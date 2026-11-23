import { fireEvent, render, screen } from "@testing-library/react";
import { Select } from "./Select";

const options = [
  { value: "car_1", label: "car_1" },
  { value: "pets_1", label: "pets_1" },
  { value: "world_1", label: "world_1", detail: "4 tables" },
];

describe("Select", () => {
  it("opens on click and picks an option", () => {
    const onChange = vi.fn();
    render(<Select aria-label="Database" options={options} value="car_1" onChange={onChange} />);
    const button = screen.getByRole("button", { name: "Database" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "car_1" })).toHaveAttribute("aria-selected", "true");
    fireEvent.click(screen.getByRole("option", { name: /world_1/ }));
    expect(onChange).toHaveBeenCalledWith("world_1");
  });

  it("works from the keyboard, with type-ahead", () => {
    const onChange = vi.fn();
    render(<Select aria-label="Database" options={options} value="car_1" onChange={onChange} />);
    fireEvent.keyDown(screen.getByRole("button", { name: "Database" }), { key: "ArrowDown" });
    const list = screen.getByRole("listbox");
    fireEvent.keyDown(list, { key: "p" });
    fireEvent.keyDown(list, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith("pets_1");
  });

  it("closes on Escape without changing the value", () => {
    const onChange = vi.fn();
    render(<Select aria-label="Database" options={options} value="car_1" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Database" }));
    fireEvent.keyDown(screen.getByRole("listbox"), { key: "Escape" });
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Database" })).toHaveAttribute("aria-expanded", "false");
  });
});
