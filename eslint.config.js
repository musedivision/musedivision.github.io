import js from '@eslint/js';
import globals from 'globals';

const rules = {
  eqeqeq: ['error', 'smart'],
  'prefer-const': 'error',
  'no-var': 'error',
  'no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
};

export default [
  { ignores: ['dist/**'] },
  js.configs.recommended,
  {
    // Browser code — the site itself.
    files: ['src/**/*.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: globals.browser,
    },
    rules,
  },
  {
    // Node code — build scripts and config.
    files: ['scripts/**/*.js', 'vite.config.js', 'eslint.config.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: globals.node,
    },
    rules,
  },
];
