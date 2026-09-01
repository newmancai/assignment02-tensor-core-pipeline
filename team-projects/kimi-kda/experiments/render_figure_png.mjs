#!/usr/bin/env node

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const sharp = require("sharp");
const root = path.dirname(fileURLToPath(import.meta.url));
const input = path.join(root, "figures", "kimi_kda_b300_bottleneck.svg");
const output = path.join(root, "figures", "kimi_kda_b300_bottleneck.png");

await sharp(input).png().toFile(output);
console.log(`wrote ${output}`);
