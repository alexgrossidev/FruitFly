// Validate replay JSON files with the viewer's own validator and atlas.
//
// This runs exactly the code path the "Load model JSON" button runs in the
// browser: viewer/src/lib/replay.ts's parseReplay(), against the set of visible
// body IDs decoded from viewer/public/data/brain-atlas/{ids,groups}.bin the same
// way viewer/src/lib/atlas.ts decodes them (groups < 3). It also enforces the
// 10 MB upload limit from viewer/src/App.tsx.
//
//   node scripts/validate_replay.mjs results/replays/*.json
//
// Node >= 22.18 (type stripping for the .ts import). Exit code 1 on any failure.
import { readFileSync, statSync } from 'node:fs';
import { parseReplay } from '../viewer/src/lib/replay.ts';

const root = new URL('../viewer/public/data/brain-atlas/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', root), 'utf8'));
const ids = readFileSync(new URL('ids.bin', root));
const groups = readFileSync(new URL('groups.bin', root));
const visible = new Set();
for (let i = 0; i < groups.length; i++) if (groups[i] < 3) visible.add(ids.readUInt32LE(i * 4));
if (visible.size !== manifest.brainCount) throw Error('Brain selection does not match the manifest.');

const files = process.argv.slice(2);
if (files.length === 0) { console.error('usage: node scripts/validate_replay.mjs <replay.json>...'); process.exit(2); }
let failed = 0;
for (const file of files) {
  try {
    const size = statSync(file).size;
    if (size > 10 * 1024 * 1024) throw Error('Replay must be under 10 MB.');
    const replay = parseReplay(JSON.parse(readFileSync(file, 'utf8')), visible);
    const idsUsed = new Set(replay.frames.flatMap(f => f.values.map(v => v[0])));
    console.log(`ok    ${file}  frames=${replay.frames.length}  ids=${idsUsed.size}  kind=${replay.source.kind}  ${(size / 1024).toFixed(1)} KiB`);
  } catch (error) {
    failed++;
    console.log(`FAIL  ${file}  ${error.message ?? error}`);
  }
}
console.log(`${files.length - failed}/${files.length} replay files valid against ${visible.size.toLocaleString('en-US')} visible MaleCNS body IDs`);
process.exit(failed ? 1 : 0);
