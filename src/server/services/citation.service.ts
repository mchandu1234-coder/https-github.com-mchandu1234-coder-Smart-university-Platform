import { VectorPayload } from "./vector.service";
import { FusedChunk } from "./fusion.service";

export interface Citation {
  citationId: string;
  id: number;
  documentId: string;
  documentName: string;
  documentVersion: number;
  docHash?: string;
  chunkId: string;
  chunkHash?: string;
  chunkIndex: number;
  pageNumber: number | null;
  sectionHeader?: string | null;
  organizationId: string;
  confidence: number;
  retrievalMethod: string;
  content: string;
}

export class CitationService {
  /**
   * Formats the compressed payloads into a strict citation block for the LLM prompt.
   * Includes document version, page number, chunk index, and section header.
   */
  static formatCitations(chunks: VectorPayload[]): string {
    if (chunks.length === 0) return "No context provided.";

    return chunks.map((chunk, index) => {
      const docVer = chunk.documentVersion || chunk.version || 1;
      const pageInfo = chunk.pageNumber || chunk.metadata?.pageNumber
        ? ` | Page: ${chunk.pageNumber || chunk.metadata?.pageNumber}`
        : "";
      const sectionInfo = chunk.sectionHeader || chunk.metadata?.title
        ? ` | Section: ${chunk.sectionHeader || chunk.metadata?.title}`
        : "";
      const chunkType = chunk.metadata?.chunkType;
      const typeInfo = chunkType && chunkType !== "text"
        ? ` | Element Type: ${chunkType.toUpperCase()}`
        : "";
      const visualInfo = chunk.metadata?.visualContext
        ? `\nVisual Context: ${chunk.metadata.visualContext}`
        : "";
      const summaryInfo = chunk.metadata?.summary
        ? `\nElement Summary: ${chunk.metadata.summary}`
        : "";
      const confidenceInfo = chunk.metadata?.fusionScore
        ? ` | Confidence: ${(chunk.metadata.fusionScore * 100).toFixed(1)}%`
        : "";

      return `[Citation ID: ${index + 1}]
Source: ${chunk.documentName || "Unknown Document"} (v${docVer}, Chunk: ${chunk.chunkIndex})${pageInfo}${sectionInfo}${typeInfo}${confidenceInfo}${visualInfo}${summaryInfo}
Document ID: ${chunk.documentId || "N/A"} | Chunk Hash: ${chunk.chunkHash || chunk.chunkId}
Content:
${chunk.chunkText}`;
    }).join("\n\n------------------------\n\n");
  }

  /**
   * Formats FusedChunk results into structured citation objects
   * for the frontend debug panel and API responses.
   */
  static formatStructuredCitations(chunks: FusedChunk[]): Citation[] {
    return chunks.map((chunk, index) => {
      let method = "hybrid";
      if (chunk.vectorScore !== null && chunk.bm25Score === null) method = "vector";
      if (chunk.vectorScore === null && chunk.bm25Score !== null) method = "bm25";

      const docVer = (chunk as any).documentVersion || (chunk as any).version || 1;

      return {
        citationId: `Citation ${index + 1}`,
        id: index + 1,
        documentId: chunk.documentId,
        documentName: chunk.documentName || "Unknown Document",
        documentVersion: docVer,
        docHash: (chunk as any).docHash,
        chunkId: chunk.chunkId,
        chunkHash: (chunk as any).chunkHash,
        chunkIndex: chunk.chunkIndex,
        pageNumber: chunk.pageNumber,
        sectionHeader: (chunk as any).sectionHeader || null,
        organizationId: chunk.organizationId,
        confidence: chunk.fusionScore,
        retrievalMethod: method,
        content: chunk.chunkText,
      };
    });
  }
}


