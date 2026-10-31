import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createGoogle } from "@ai-sdk/google";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ModelConfig } from "@/ai/llm/model-config";
import { extractTextWithPages } from "./pdf-loader";

// Lazily initialize OpenRouter client to prevent static import initialization timing issues
let openrouterInstance: ReturnType<typeof createOpenAI> | null = null;

function getOpenRouter() {
  if (!openrouterInstance) {
    openrouterInstance = createOpenAI({
      baseURL: ModelConfig.baseUrl,
      apiKey: process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || "",
    });
  }
  return openrouterInstance;
}

// Lazily initialize Google Gemini client
let googleInstance: ReturnType<typeof createGoogle> | null = null;

function getGoogleProvider() {
  if (!googleInstance) {
    googleInstance = createGoogle({
      apiKey: process.env.GEMINI_API_KEY || "",
    });
  }
  return googleInstance;
}

export interface ParsedElement {
  type: "text" | "table" | "chart" | "image";
  content: string; // Plain text, Markdown table, or chart summary text
  pageNumber: number;
  metadata: {
    title?: string;
    headers?: string[];
    confidence?: number;
    visualContext?: string; // For charts: description of trends, axis labels, legends
    summary?: string; // Short 1-2 sentence semantic summary of this element for vector search indexing
  };
}

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,
  chunkOverlap: 150,
});

/**
 * Extracts and validates JSON elements array from multimodal model output.
 * Strips reasoning preambles, markdown formatting fences, and trailing commentary.
 */
function parseAndValidateJsonElements(textOutput: string): ParsedElement[] {
  if (!textOutput || !textOutput.trim()) return [];

  // Extract content between outermost brackets to safely strip preamble or conversational commentary
  const firstBracket = textOutput.indexOf("[");
  const lastBracket = textOutput.lastIndexOf("]");

  let cleanJson = "";
  if (firstBracket !== -1 && lastBracket !== -1 && lastBracket > firstBracket) {
    cleanJson = textOutput.substring(firstBracket, lastBracket + 1);
  } else {
    cleanJson = textOutput
      .replace(/```json/gi, "")
      .replace(/```/g, "")
      .trim();
  }

  try {
    const parsed = JSON.parse(cleanJson);
    if (Array.isArray(parsed) && parsed.length > 0) {
      return parsed
        .filter(
          (el) =>
            el &&
            typeof el === "object" &&
            typeof el.content === "string" &&
            el.content.trim().length > 0
        )
        .map((el) => ({
          type: (["text", "table", "chart", "image"].includes(el.type)
            ? el.type
            : "text") as ParsedElement["type"],
          content: el.content.trim(),
          pageNumber: Number(el.pageNumber) || 1,
          metadata: {
            title: el.metadata?.title || undefined,
            headers: Array.isArray(el.metadata?.headers) ? el.metadata.headers : undefined,
            visualContext: el.metadata?.visualContext || undefined,
            summary: el.metadata?.summary || undefined,
          },
        }));
    }
  } catch (parseErr) {
    console.warn("[LayoutLoader] JSON.parse error on multimodal model output:", parseErr);
  }

  return [];
}

/**
 * Parses a PDF natively using Gemini 2.5 Flash Vision with automatic resilient fallback.
 * Extracts paragraphs, structured Markdown tables, chart descriptions, and scanned OCR elements.
 */
export async function parsePdfNativeMultimodal(
  pdfBuffer: Buffer,
  fileName: string
): Promise<ParsedElement[]> {
  const base64Pdf = pdfBuffer.toString("base64");

  const prompt = `
You are a layout-aware PDF document parser. Analyze the uploaded PDF document named "${fileName}".
Understand its layout, separating standard text paragraphs from tables, charts, scanned pages, and images.

Process the document and extract elements:
1. **Text**: Extract paragraphs. Keep headings attached.
2. **Tables**: Render all tables as structured Markdown tables, preserving row and column alignments.
3. **Charts & Graphs**: Write a detailed textual summary of the chart, including axis legends, labels, visual trends, data points, and the overall conclusion.
4. **Scanned Images / Pages**: Perform OCR to extract text structures.

Output the result as a single, valid JSON array containing objects matching this format:
\`\`\`json
[
  {
    "type": "text" | "table" | "chart" | "image",
    "content": "the extracted text or markdown table or visual summary",
    "pageNumber": 1,
    "metadata": {
      "title": "optional title or header of the element",
      "headers": ["col1", "col2"], // ONLY for table elements
      "visualContext": "axes, trends, and data point summaries", // ONLY for chart elements
      "summary": "a short 1-2 sentence semantic summary of this element for vector search indexing" // REQUIRED for table/chart/image
    }
  }
]
\`\`\`
Return ONLY the raw JSON array. Do not include markdown formatting wraps, fences (like \`\`\`json), or any introductory text.
`;

  // 1. Primary Engine: Direct Google Gemini 2.5 Flash Vision
  // Provides native PDF understanding, table parsing, chart reading, and OCR with no file-balance limitations.
  if (process.env.GEMINI_API_KEY) {
    try {
      console.log(
        `[LayoutLoader] Executing Gemini Multimodal Vision for "${fileName}" (${(
          pdfBuffer.length / 1024
        ).toFixed(1)} KB)...`
      );
      const response = await generateText({
        model: getGoogleProvider()(ModelConfig.multimodalDirect),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "file",
                data: base64Pdf,
                mediaType: "application/pdf",
              },
            ],
          },
        ],
        temperature: 0.1,
        maxOutputTokens: 16384,
      });

      const elements = parseAndValidateJsonElements(response.text);
      if (elements.length > 0) {
        console.log(
          `[LayoutLoader] Direct Gemini Multimodal parsing successful for "${fileName}": extracted ${elements.length} layout elements.`
        );
        return elements;
      }
      console.warn(
        `[LayoutLoader] Direct Gemini Multimodal returned no valid elements for "${fileName}". Trying secondary engine...`
      );
    } catch (geminiErr: any) {
      console.warn(
        `[LayoutLoader] Direct Gemini Multimodal parsing failed for "${fileName}":`,
        geminiErr?.message || geminiErr
      );
    }
  }

  // 2. Secondary Engine: OpenRouter Multimodal Vision (if Gemini key not present or failed)
  if (process.env.OPENROUTER_API_KEY) {
    try {
      console.log(`[LayoutLoader] Attempting OpenRouter Multimodal Vision for "${fileName}"...`);
      const response = await generateText({
        model: getOpenRouter().chat(ModelConfig.multimodal),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "file",
                data: base64Pdf,
                mediaType: "application/pdf",
              },
            ],
          },
        ],
        temperature: 0.1,
        maxOutputTokens: 16384,
      });

      const elements = parseAndValidateJsonElements(response.text);
      if (elements.length > 0) {
        console.log(
          `[LayoutLoader] OpenRouter Multimodal parsing successful for "${fileName}": extracted ${elements.length} layout elements.`
        );
        return elements;
      }
    } catch (openRouterErr: any) {
      console.warn(
        `[LayoutLoader] OpenRouter Multimodal parsing failed for "${fileName}":`,
        openRouterErr?.message || openRouterErr
      );
    }
  }

  // 3. Resilient Fallback: Local text extraction (for offline development or API outages)
  console.warn(
    `[LayoutLoader] Multimodal engines unavailable or failed for "${fileName}". Falling back to resilient local text extraction.`
  );
  const { pages, fullText } = await extractTextWithPages(pdfBuffer);
  if (pages && pages.length > 0) {
    return pages.map((p) => ({
      type: "text" as const,
      content: p.text,
      pageNumber: p.pageNumber,
      metadata: {
        title: fileName,
        summary: p.text.slice(0, 200).replace(/\s+/g, " ").trim(),
      },
    }));
  }

  return [
    {
      type: "text" as const,
      content: fullText || "No extractable text found.",
      pageNumber: 1,
      metadata: {
        title: fileName,
        summary: (fullText || "").slice(0, 200),
      },
    },
  ];
}

/**
 * Splits extracted layout elements into chunks optimized for embeddings.
 * Retains table structures and chart summaries intact, chunking only large text segments.
 */
export async function chunkParsedElements(
  elements: ParsedElement[]
): Promise<Array<{ text: string; pageNumber: number; metadata: any }>> {
  const chunks: Array<{ text: string; pageNumber: number; metadata: any }> = [];

  for (const element of elements) {
    // Keep tables and charts whole to avoid splitting relational structures
    if (element.type === "table" || element.type === "chart" || element.type === "image") {
      chunks.push({
        text: element.content,
        pageNumber: element.pageNumber,
        metadata: {
          chunkType: element.type,
          title: element.metadata?.title || null,
          tableHeaders: element.metadata?.headers || [],
          visualContext: element.metadata?.visualContext || null,
          summary: element.metadata?.summary || null,
        },
      });
    } else {
      // Split large text paragraphs semantically
      const splitTexts = await textSplitter.splitText(element.content);
      splitTexts.forEach((text) => {
        if (text && text.trim().length > 0) {
          chunks.push({
            text: text.trim(),
            pageNumber: element.pageNumber,
            metadata: {
              chunkType: "text",
              title: element.metadata?.title || null,
              tableHeaders: [],
              visualContext: null,
              summary: element.metadata?.summary || null,
            },
          });
        }
      });
    }
  }

  return chunks;
}
