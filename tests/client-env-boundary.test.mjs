import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const viteConfig = await readFile(new URL('../vite.config.ts', import.meta.url), 'utf8');
const readme = await readFile(new URL('../README.md', import.meta.url), 'utf8');
const gitignore = await readFile(new URL('../.gitignore', import.meta.url), 'utf8');

test('Vite does not load or inject provider credentials into browser code', () => {
  assert.doesNotMatch(viteConfig, /\bloadEnv\b/);
  assert.doesNotMatch(viteConfig, /process\.env\.(?:API_KEY|GEMINI_API_KEY)/);
  assert.doesNotMatch(viteConfig, /GEMINI_API_KEY/);
});

test('setup instructions do not ask users to place provider keys in the client environment', () => {
  assert.doesNotMatch(readme, /set\s+the\s+`?GEMINI_API_KEY/i);
  assert.match(readme, /Do not put provider credentials in `\.env\.local`/);
});

test('tracked local environment files are absent and ignored', async () => {
  await assert.rejects(
    readFile(new URL('../.env.local', import.meta.url)),
    { code: 'ENOENT' },
  );
  assert.match(gitignore, /^\*\.local$/m);
});
