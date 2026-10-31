import { db } from "@/server/db/prisma";
import { createClient } from "@/utils/insforge/server";
import { extractTextFromPdf, chunkText, extractTextWithPages, chunkTextWithMetadata } from "@/ai/loaders/pdf-loader";
import { extractTextFromDocx, extractTextWithPagesDocx } from "@/ai/loaders/docx-loader";
import { extractTextFromTxt, extractTextWithPagesTxt } from "@/ai/loaders/txt-loader";
import { parsePdfNativeMultimodal, chunkParsedElements } from "@/ai/loaders/layout-loader";
import { ensureCollectionExists } from "@/ai/vector/qdrant";
import { EmbeddingService } from "./embedding.service";
import { VectorService, VectorPayload } from "./vector.service";
import { GraphExtractionService } from "./graph-extraction.service";
import { v4 as uuidv4 } from "uuid";
import tiktoken from "tiktoken";
import { AuditService } from "./audit";
import { generateDocumentHash, generateChunkHash } from "@/server/utils/chunk-hasher";
 
export class DocumentService {
  /**
   * Main entry point to upload a document to InsForge storage (Synchronous).
   * It creates a DB record with PROCESSING status and returns early.
   */
  static async uploadDocument(
    file: File,
    organizationId: string,
    uploadedBy: string,
    knowledgeBaseId?: string,
    departmentId?: string | null,
    collegeId?: string | null,
    visibility: import("@prisma/client").DocumentVisibility = "DEPARTMENT"
  ) {
    const insforge = await createClient();
    
    const fileExt = file.name.split('.').pop();
    const fileName = `${uuidv4()}.${fileExt}`;
    const storagePath = `${organizationId}/${fileName}`;

    // 1. Upload to InsForge Storage (bucket: 'documents')
    const { error: uploadError } = await insforge.storage
      .from('documents')
      .upload(storagePath, file);

    if (uploadError) {
      throw new Error(`Failed to upload to InsForge: ${uploadError.message}`);
    }

    // 2. Create Document record in Prisma
    const document = await db.document.create({
      data: {
        fileName: file.name,
        fileType: file.type,
        fileSize: file.size,
        storagePath: storagePath,
        organizationId,
        knowledgeBaseId, // Optional KB relation
        uploadedBy,
        departmentId: departmentId || null,
        collegeId: collegeId || null,
        visibility,
        processingStatus: "PROCESSING",
      },
    });

    // Log the upload event
    await AuditService.logEvent({
      orgId: organizationId,
      userId: uploadedBy,
      action: "DOCUMENT_UPLOAD",
      metadata: {
        documentId: document.id,
        fileName: document.fileName,
        fileSize: document.fileSize,
        knowledgeBaseId,
        departmentId,
        collegeId,
        visibility,
      },
    });
    
    return document;
  }

  /**
   * Detects the document type from the file name extension.
   * Returns a normalized type string: 'pdf', 'docx', 'txt', or 'unknown'.
   */
  private static detectFileType(fileName?: string, fileType?: string): "pdf" | "docx" | "txt" | "unknown" {
    const ext = fileName?.split(".").pop()?.toLowerCase();
    if (ext === "pdf" || fileType === "application/pdf") return "pdf";
    if (
      ext === "docx" ||
      fileType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
      return "docx";
    if (ext === "txt" || fileType === "text/plain") return "txt";
    return "unknown";
  }

  /**
   * Background process for text extraction, chunking, and embedding.
   */
  static async processDocumentAsync(documentId: string, fileBuffer: ArrayBuffer, organizationId: string, knowledgeBaseId?: string, fileName?: string) {
    const log = (msg: string) => console.log(`[DocProcess:${documentId.slice(0,8)}] ${msg}`);
    try {
      log(`Starting processing for: ${fileName}`);
      const buffer = Buffer.from(fileBuffer);

      // Detect file type from filename or DB record
      const docRecord = await db.document.findUnique({ where: { id: documentId } });
      const detectedType = DocumentService.detectFileType(fileName, docRecord?.fileType);
      log(`Detected file type: ${detectedType}`);

      // Step 1: Extract text and layout elements based on file type
      log("Step 1: Extracting text and layout elements...");
      let text = "";
      let layoutChunks: Array<{ text: string; pageNumber: number | null; metadata: any }> = [];

      if (detectedType === "pdf") {
        log("Parsing PDF using Layout-Aware Multimodal Gemini Vision...");
        const parsedElements = await parsePdfNativeMultimodal(buffer, fileName || "document.pdf");
        
        layoutChunks = await chunkParsedElements(parsedElements);
        
        text = parsedElements
          .map((el) => `[Page ${el.pageNumber} - ${el.type.toUpperCase()}]\n${el.content}`)
          .join("\n\n");
          
        log(`Layout extraction complete. Extracted ${parsedElements.length} elements, split into ${layoutChunks.length} chunks.`);
      } else {
        let extractionResult: {
          pages: Array<{ pageNumber: number; text: string }>;
          fullText: string;
          numPages: number;
        };

        switch (detectedType) {
          case "docx":
            extractionResult = await extractTextWithPagesDocx(buffer);
            break;
          case "txt":
            extractionResult = await extractTextWithPagesTxt(buffer);
            break;
          default:
            log(`Unknown file type, falling back to PDF extraction`);
            extractionResult = await extractTextWithPages(buffer);
            break;
        }

        const { pages, fullText, numPages } = extractionResult;
        text = fullText;
        log(`Step 1 done. Text length: ${text.length} chars, Pages: ${numPages}`);
        
        log("Step 2: Chunking text with page metadata...");
        const chunksWithMeta = await chunkTextWithMetadata(text, pages);
        layoutChunks = chunksWithMeta.map((c) => ({
          text: c.text,
          pageNumber: c.pageNumber,
          metadata: {
            chunkType: "text",
            title: null,
            tableHeaders: [],
            visualContext: null,
          },
        }));
      }

      if (!text || text.trim().length === 0) {
        throw new Error("Document appears to be empty or contains no extractable text.");
      }

      await db.document.update({
        where: { id: documentId },
        data: { content: text },
      });

      log(`Step 2: Chunk parsing completed. Total chunks: ${layoutChunks.length}`);
      if (layoutChunks.length === 0) {
        throw new Error("No chunks produced from document.");
      }

      // Step 3: Ensure Qdrant collection
      log("Step 3: Ensuring Qdrant collection exists...");
      await ensureCollectionExists();
      log("Step 3 done.");

      // Step 4: Embed + store
      const BATCH_SIZE = 50;
      const encoder = tiktoken.encoding_for_model("text-embedding-3-small");

      const docHash = generateDocumentHash(buffer);
      const documentVersion = 1; // Default to canonical v1 for newly ingested documents

      // Clean up previous chunks for idempotent re-processing
      await db.chunk.deleteMany({ where: { documentId } });

      for (let i = 0; i < layoutChunks.length; i += BATCH_SIZE) {
        const batchChunks = layoutChunks.slice(i, i + BATCH_SIZE);
        
        // Parent-Child Multi-Vector: Embed semantic summaries for tables/charts/images, keep raw content as target
        const batchTexts = batchChunks.map((c) => {
          if (
            (c.metadata?.chunkType === "table" ||
              c.metadata?.chunkType === "chart" ||
              c.metadata?.chunkType === "image") &&
            c.metadata?.summary
          ) {
            const titlePrefix = c.metadata.title ? `Document element: ${c.metadata.title}. ` : "";
            return `${titlePrefix}Summary: ${c.metadata.summary}`;
          }
          return c.text;
        });

        log(`Step 4: Embedding batch ${i / BATCH_SIZE + 1} (${batchTexts.length} chunks)...`);

        const vectors = await EmbeddingService.embedBatch(batchTexts, 3);

        if (vectors.length !== batchTexts.length) {
          throw new Error("Mismatch between batch size and generated vectors.");
        }

        const qdrantPoints = [];

        for (let j = 0; j < batchChunks.length; j++) {
          const chunk = batchChunks[j];
          const chunkTextContent = chunk.text;
          const globalIndex = i + j;
          const tokenCount = encoder.encode(chunkTextContent).length;

          const { chunkHash, chunkId } = generateChunkHash(
            documentId,
            documentVersion,
            globalIndex,
            chunkTextContent
          );

          const chunkRecord = await db.chunk.create({
            data: {
              content: chunkTextContent, // Full parent Markdown table / visual context
              chunkIndex: globalIndex,
              tokenCount,
              documentId,
              organizationId,
              pageNumber: chunk.pageNumber,
              metadata: {
                pageNumber: chunk.pageNumber,
                chunkType: chunk.metadata?.chunkType || "text",
                title: chunk.metadata?.title || null,
                tableHeaders: chunk.metadata?.tableHeaders || [],
                visualContext: chunk.metadata?.visualContext || null,
                summary: chunk.metadata?.summary || null,
                docHash,
                chunkHash,
              },
            },
          });

          const payload: VectorPayload = {
            organizationId,
            collegeId: docRecord?.collegeId || null,
            departmentId: docRecord?.departmentId || null,
            visibility: docRecord?.visibility || "DEPARTMENT",
            uploadedBy: docRecord?.uploadedBy || null,
            documentId,
            documentName: fileName || "Unknown",
            documentVersion,
            version: documentVersion,
            isLatest: true,
            docHash,
            chunkId,
            chunkHash,
            chunkIndex: globalIndex,
            chunkText: chunkTextContent,
            pageNumber: chunk.pageNumber || null,
            sectionHeader: chunk.metadata?.title || null,
            createdAt: new Date().toISOString(),
            ...(knowledgeBaseId && { knowledgeBaseId }),
            metadata: {
              pageNumber: chunk.pageNumber,
              chunkType: chunk.metadata?.chunkType || "text",
              title: chunk.metadata?.title || null,
              tableHeaders: chunk.metadata?.tableHeaders || [],
              visualContext: chunk.metadata?.visualContext || null,
              summary: chunk.metadata?.summary || null,
              collegeId: docRecord?.collegeId || null,
              departmentId: docRecord?.departmentId || null,
              visibility: docRecord?.visibility || "DEPARTMENT",
              uploadedBy: docRecord?.uploadedBy || null,
            },
          };

          qdrantPoints.push({ id: chunkId, vector: vectors[j], payload });
        }

        await VectorService.upsertBatch(qdrantPoints);
        log(`Batch ${i / BATCH_SIZE + 1} upserted to Qdrant.`);
      }

      encoder.free();

      // Step 5: Graph extraction (non-blocking)
      log("Step 5: Triggering graph extraction (non-blocking)...");
      GraphExtractionService.extractAndIngest(text, organizationId, documentId).catch((err) => {
        log(`Graph extraction failed (non-fatal): ${err?.message ?? err}`);
      });

      // Mark completed
      await db.document.update({
        where: { id: documentId },
        data: { processingStatus: "COMPLETED" },
      });
      log("✅ Processing COMPLETED");

      // Log success event (non-blocking)
      const doc = await db.document.findUnique({ where: { id: documentId } });
      AuditService.logEvent({
        orgId: organizationId,
        userId: doc?.uploadedBy,
        action: "DOCUMENT_PROCESSING_COMPLETED",
        metadata: {
          documentId,
          fileName: fileName || doc?.fileName,
          chunksCount: layoutChunks.length,
        },
      }).catch(() => {});

    } catch (error: any) {
      console.error(`[DocProcess:${documentId.slice(0,8)}] ❌ FAILED:`, error?.message ?? error);
      await db.document.update({
        where: { id: documentId },
        data: { processingStatus: "FAILED" },
      });

      // Log failure event (non-blocking)
      const doc = await db.document.findUnique({ where: { id: documentId } });
      AuditService.logEvent({
        orgId: organizationId,
        userId: doc?.uploadedBy,
        action: "DOCUMENT_PROCESSING_FAILED",
        metadata: {
          documentId,
          fileName: fileName || doc?.fileName,
          error: error?.message || String(error),
        },
      }).catch(() => {});

      throw error;
    }
  }

  /**
   * Reprocesses an existing document from InsForge storage.
   */
  static async reprocessDocument(documentId: string) {
    const doc = await db.document.findUnique({ where: { id: documentId } });
    if (!doc) throw new Error("Document not found");

    await db.document.update({
      where: { id: documentId },
      data: { processingStatus: "PROCESSING" },
    });

    const insforge = await createClient();
    const { data, error } = await insforge.storage
      .from("documents")
      .download(doc.storagePath);

    if (error || !data) {
      await db.document.update({
        where: { id: documentId },
        data: { processingStatus: "FAILED" },
      });
      throw new Error(`Failed to download document from storage: ${error?.message || "No data"}`);
    }

    const fileBuffer = await data.arrayBuffer();
    return DocumentService.processDocumentAsync(
      doc.id,
      fileBuffer,
      doc.organizationId,
      doc.knowledgeBaseId || undefined,
      doc.fileName
    );
  }
}
