import { render, screen } from "@testing-library/react";
import { ResultTable } from "./ResultTable";
import type { Execution } from "../api/types";

const base: Execution = {
  status: "ok",
  columns: ["name", "age"],
  rows: [
    ["Joe", 52],
    [null, 41],
  ],
  row_count: 2,
  truncated: false,
  error: null,
};

test("renders a real table with numbers right-aligned and NULL greyed", () => {
  render(<ResultTable execution={base} />);
  expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["name", "age"]);
  expect(screen.getByText("52")).toHaveClass("num");
  expect(screen.getByText("Joe")).not.toHaveClass("num");
  expect(screen.getByText("NULL")).toHaveClass("null");
  expect(screen.queryByText(/Showing the first/)).toBeNull();
});

test("truncated results say so", () => {
  render(<ResultTable execution={{ ...base, truncated: true, row_count: 500 }} />);
  expect(screen.getByText("Showing the first 500 rows.")).toBeInTheDocument();
});

test("zero rows is a valid answer, not an error", () => {
  render(<ResultTable execution={{ ...base, rows: [], row_count: 0 }} />);
  expect(screen.getByText("The query ran but no rows matched.")).toBeInTheDocument();
  expect(screen.queryByRole("table")).toBeNull();
});
