#!/usr/bin/env node
/** Build a score-only supervision variant while preserving prompts and sample splits. */

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

function parseArguments(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(
        "Usage: make_score_only.js --input FILE --output FILE --source-split FILE --output-split FILE",
      );
    }
    parsed[key.slice(2)] = value;
  }
  for (const key of ["input", "output", "source-split", "output-split"]) {
    if (!parsed[key]) throw new Error(`Missing --${key}`);
  }
  return parsed;
}

function sha256(content) {
  return crypto.createHash("sha256").update(content).digest("hex");
}

function writeAtomically(filename, content) {
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.tmp`;
  fs.writeFileSync(temporary, content);
  fs.renameSync(temporary, filename);
}

const args = parseArguments(process.argv.slice(2));
const inputPath = path.resolve(args.input);
const outputPath = path.resolve(args.output);
const sourceSplitPath = path.resolve(args["source-split"]);
const outputSplitPath = path.resolve(args["output-split"]);

if (inputPath === outputPath || sourceSplitPath === outputSplitPath) {
  throw new Error("Output paths must not overwrite the full-CoT dataset or its split");
}

const inputContent = fs.readFileSync(inputPath, "utf8");
const inputHash = sha256(inputContent);
const inputLines = inputContent.trimEnd().split("\n");
const seenIds = new Set();
const labelCounts = {};

const outputRows = inputLines.map((line, index) => {
  const row = JSON.parse(line);
  const lineNumber = index + 1;
  if (typeof row.id !== "string" || !row.id || seenIds.has(row.id)) {
    throw new Error(`${inputPath}:${lineNumber} has an invalid or duplicate id`);
  }
  if (!Number.isInteger(row.label)) {
    throw new Error(`${inputPath}:${lineNumber} has an invalid label`);
  }
  if (
    !Array.isArray(row.completion) ||
    row.completion.length !== 1 ||
    row.completion[0]?.role !== "assistant" ||
    typeof row.completion[0]?.content !== "string"
  ) {
    throw new Error(`${inputPath}:${lineNumber} has an invalid completion`);
  }

  const matches = [
    ...row.completion[0].content.matchAll(/<score>\s*(-?\d+)\s*<\/score>/gi),
  ];
  if (matches.length !== 1 || Number(matches[0][1]) !== row.label) {
    throw new Error(`${inputPath}:${lineNumber} score does not match label`);
  }

  seenIds.add(row.id);
  labelCounts[row.label] = (labelCounts[row.label] || 0) + 1;
  return {
    ...row,
    completion: [
      {
        ...row.completion[0],
        content: `<score>${row.label}</score>`,
      },
    ],
  };
});

const outputContent = `${outputRows.map(JSON.stringify).join("\n")}\n`;
const outputHash = sha256(outputContent);
const sourceSplit = JSON.parse(fs.readFileSync(sourceSplitPath, "utf8"));
if (sourceSplit.dataset_sha256 !== inputHash) {
  throw new Error("Source split hash does not match the full-CoT dataset");
}

const trainIds = sourceSplit.train_ids || [];
const validationIds = sourceSplit.validation_ids || [];
const splitIds = [...trainIds, ...validationIds];
if (
  splitIds.length !== seenIds.size ||
  new Set(splitIds).size !== splitIds.length ||
  splitIds.some((id) => !seenIds.has(id))
) {
  throw new Error("Source split does not exactly cover the dataset IDs");
}

const outputSplit = {
  ...sourceSplit,
  dataset_sha256: outputHash,
};
writeAtomically(outputPath, outputContent);
writeAtomically(outputSplitPath, `${JSON.stringify(outputSplit, null, 2)}\n`);

console.log(
  JSON.stringify(
    {
      input: inputPath,
      output: outputPath,
      samples: outputRows.length,
      labels: labelCounts,
      input_sha256: inputHash,
      output_sha256: outputHash,
      train_samples: trainIds.length,
      validation_samples: validationIds.length,
      prompt_and_metadata_preserved: true,
      completion_template: "<score>{label}</score>",
    },
    null,
    2,
  ),
);
