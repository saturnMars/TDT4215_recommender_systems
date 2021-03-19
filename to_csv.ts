// 1. Install the Deno runtime from https://deno.land
// 2. Run this file:
// deno run -A to_csv.ts

// You may need to adjust the data directory:
const dataDirectory = "/tmp/home/lemeiz/content_refine/";

const decoder = new TextDecoder("utf-8");
const fields = ["id", "keywords", "title", "createtime"];

async function print(file: string) {
  const data = await Deno.readFile(dataDirectory + file);
  const decoded = decoder.decode(data);
  const json = JSON.parse(decoded);
  const values = fields
    .map((f) => {
      const v = json.fields.filter((e: { field: string }) => e.field === f);
      return v.length === 0 ? "" : v[0].value;
    })
    .map((v: string) => `"${v.split('"').join('""')}"`)
    .join(",");
  console.log(values);
}

console.log('"id", "keywords", "title", "createtime"');
for await (const file of Deno.readDir(dataDirectory)) {
  await print(file.name);
}
