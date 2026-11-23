import "@testing-library/jest-dom/vitest";

// jsdom has no layout; CodeMirror measures text ranges, so give it empty rects.
const emptyRects = () => Object.assign([], { item: () => null }) as unknown as DOMRectList;
Range.prototype.getClientRects = emptyRects;
Range.prototype.getBoundingClientRect = () => new DOMRect();

// jsdom has no scrollIntoView (the dropdown keeps its active option in view)
Element.prototype.scrollIntoView ??= function () {};
