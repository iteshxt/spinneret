import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SchemaExplorer } from "./SchemaExplorer";
import type { SchemaInfo, SchemaItems } from "../api/types";

const schema: SchemaInfo = {
  db_id: "concert_singer",
  tables: [
    { name: "stadium", columns: [{ name: "Stadium_ID", type: "number", is_pk: true }] },
    {
      name: "singer",
      columns: [
        { name: "Singer_ID", type: "number", is_pk: true },
        { name: "Country", type: "text", is_pk: false },
        { name: "Age", type: "number", is_pk: false },
      ],
    },
  ],
  foreign_keys: [],
};

const focus: SchemaItems = {
  tables: [
    { name: "singer", score: 0.99, kept: true },
    { name: "stadium", score: 0.03, kept: false },
  ],
  columns: [
    { table: "singer", name: "Country", score: 0.98, kept: true },
    { table: "singer", name: "Age", score: 0.1, kept: false },
  ],
};

test("without focus, tables are collapsed and unscored", () => {
  render(<SchemaExplorer schema={schema} focus={null} />);
  expect(screen.getByRole("button", { name: "Table stadium" })).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText(".99")).toBeNull();
});

test("focus mode highlights kept items, orders by score and says so in text", () => {
  render(<SchemaExplorer schema={schema} focus={focus} />);
  const tables = screen.getAllByRole("button", { name: /^Table / });
  expect(tables[0]).toHaveAccessibleName("Table singer, kept by model, score .99");
  expect(tables[0]).toHaveClass("kept-bar");
  expect(tables[0]).toHaveAttribute("aria-expanded", "true"); // kept tables auto-expand
  expect(tables[1]).toHaveClass("not-kept");
  expect(screen.getByRole("button", { name: /Column Country.*kept by model/ })).toHaveClass("kept");
  expect(screen.getByRole("button", { name: /Column Age.*filtered out by model/ })).toHaveClass("not-kept");
});

test("focus can be switched off, and clicking a column inserts it", async () => {
  const onColumn = vi.fn();
  render(<SchemaExplorer schema={schema} focus={focus} onColumn={onColumn} />);
  await userEvent.click(screen.getByRole("button", { name: /Column Country/ }));
  expect(onColumn).toHaveBeenCalledWith("Country");
  await userEvent.click(screen.getByRole("checkbox", { name: "focus" }));
  expect(screen.getByRole("button", { name: "Table singer" })).not.toHaveClass("kept-bar");
});

test("the filter keeps matching tables and opens them", async () => {
  render(<SchemaExplorer schema={schema} focus={null} />);
  await userEvent.type(screen.getByRole("textbox", { name: /Filter/ }), "count");
  expect(screen.queryByRole("button", { name: /Table stadium/ })).toBeNull();
  expect(screen.getByRole("button", { name: /Column Country/ })).toBeInTheDocument();
});
