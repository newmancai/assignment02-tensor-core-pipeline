import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import JSZip from "jszip";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = process.env.C1_DELIVERY_DIR
  ? path.resolve(process.env.C1_DELIVERY_DIR)
  : path.resolve(scriptDir, "../..");
const input = path.join(workspaceDir, "01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260911_v3.pptx");
const output = path.join(workspaceDir, "01_slides/C1_FlashKDA_SM100_奶龙必胜_公开脱敏版_20260911_v3.pptx");

const zip = await JSZip.loadAsync(await fs.readFile(input));

for (const [name, file] of Object.entries(zip.files)) {
  if (file.dir || (!name.endsWith(".xml") && !name.endsWith(".rels"))) continue;
  let xml = await file.async("string");

  // Replace repository revision identifiers that are useful internally but not needed publicly.
  xml = xml.replaceAll("1ce47ea", "版本已固定");
  xml = xml.replaceAll("5c149f5", "版本已固定");
  xml = xml.replaceAll("ChatGPT", "答辩公开版");
  xml = xml.replaceAll("奶龙必胜｜蔡雨洋 · 李奥 · 赵骋", "奶龙必胜");

  // Remove speaker notes, which contain internal evidence-file names and source navigation hints.
  if (name.startsWith("ppt/notesSlides/notesSlide") && name.endsWith(".xml")) {
    xml = xml.replace(/<a:t>[\s\S]*?<\/a:t>/g, "<a:t></a:t>");
  }

  if (name === "docProps/core.xml") {
    xml = xml
      .replace(/<dc:title>[\s\S]*?<\/dc:title>/, "<dc:title>FlashKDA SM100 迁移分析</dc:title>")
      .replace(/<dc:creator>[\s\S]*?<\/dc:creator>/, "<dc:creator>奶龙必胜</dc:creator>")
      .replace(/<cp:lastModifiedBy>[\s\S]*?<\/cp:lastModifiedBy>/, "<cp:lastModifiedBy>奶龙必胜</cp:lastModifiedBy>");
  }

  zip.file(name, xml);
}

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, await zip.generateAsync({
  type: "nodebuffer",
  compression: "DEFLATE",
  compressionOptions: { level: 9 },
}));

console.log(output);
