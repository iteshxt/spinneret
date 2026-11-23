/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" for the hosted static site: read-only data comes from /snapshot, questions go to VITE_API_URL. */
  readonly VITE_HOSTED?: string;
  /** Origin of the live server, e.g. https://laptop.tailnet.ts.net (empty: the page's own origin). */
  readonly VITE_API_URL?: string;
}
