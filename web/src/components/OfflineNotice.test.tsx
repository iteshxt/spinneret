import { fireEvent, render, screen } from "@testing-library/react";
import { OfflineNotice } from "./OfflineNotice";

describe("OfflineNotice", () => {
  it("shows a visible checking state, then the result, even when the check fails at once", async () => {
    const onCheck = vi.fn(() => Promise.reject(new Error("offline")));
    render(<OfflineNotice onCheck={onCheck} />);
    expect(screen.getByText("Checks every 30 s")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    const busy = screen.getByRole("button", { name: /Checking/ });
    expect(busy).toBeDisabled();
    expect(busy).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Contacting the laptop…")).toBeInTheDocument();
    expect(onCheck).toHaveBeenCalledTimes(1);

    expect(await screen.findByText(/Still offline · checked/, {}, { timeout: 2000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check again" })).toBeEnabled();
  });
});
