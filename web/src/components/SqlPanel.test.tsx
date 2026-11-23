import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SqlPanel } from "./SqlPanel";

const noop = () => {};

test("generated SQL shows no Edited badge", () => {
  render(
    <SqlPanel value="select 1" generated="select 1" onChange={noop} onRun={noop} onReset={noop} dark={false} />,
  );
  expect(screen.queryByText("Edited")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset" })).toBeNull();
});

test("edited SQL shows the badge, and Reset and Run call back", async () => {
  const onReset = vi.fn();
  const onRun = vi.fn();
  render(
    <SqlPanel value="select 2" generated="select 1" onChange={noop} onRun={onRun} onReset={onReset} dark={false} />,
  );
  expect(screen.getByText("Edited")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Reset" }));
  await userEvent.click(screen.getByRole("button", { name: /Run/ }));
  expect(onReset).toHaveBeenCalledOnce();
  expect(onRun).toHaveBeenCalledOnce();
});

test("Run is disabled for empty SQL", () => {
  render(<SqlPanel value="  " generated={null} onChange={noop} onRun={noop} onReset={noop} dark />);
  expect(screen.getByRole("button", { name: /Run/ })).toBeDisabled();
});
