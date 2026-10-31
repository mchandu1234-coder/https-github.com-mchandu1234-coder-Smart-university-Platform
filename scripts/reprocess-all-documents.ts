import { db } from "../src/server/db/prisma";
import { DocumentService } from "../src/server/services/document.service";
import { loadEnvConfig } from "@next/env";

loadEnvConfig(process.cwd(), true);

async function reprocessAll() {
  console.log("=== Multimodal Document Reprocessor ===");
  const docs = await db.document.findMany({
    where: {
      deletedAt: null,
      fileType: "application/pdf",
    },
    select: {
      id: true,
      fileName: true,
      processingStatus: true,
      _count: { select: { chunks: true } },
    },
    orderBy: { createdAt: "desc" },
  });

  console.log(`Found ${docs.length} PDF documents in database.`);

  for (const doc of docs) {
    console.log(`\n--------------------------------------------------`);
    console.log(`Processing: ${doc.fileName} (${doc.id})`);
    try {
      const startTime = performance.now();
      await DocumentService.reprocessDocument(doc.id);
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      const newChunkCount = await db.chunk.count({ where: { documentId: doc.id } });
      console.log(`✅ Completed in ${elapsed}s. New chunk count: ${newChunkCount}`);
    } catch (err: any) {
      console.error(`❌ Failed to reprocess ${doc.fileName}:`, err?.message || err);
    }
  }

  console.log("\n=== All Documents Reprocessed Successfully ===");
}

reprocessAll()
  .catch(console.error)
  .finally(() => {
    db.$disconnect();
    process.exit(0);
  });
