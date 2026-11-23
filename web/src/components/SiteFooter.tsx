import { Link } from "react-router-dom";
import { LICENSE_URL, REPO } from "../site";
import { GitHubIcon } from "./icons";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-grid">
        <div className="footer-brand">
          <Link to="/" className="brand">
            <span className="brand-mark" aria-hidden>
              ◈
            </span>
            Spinneret
          </Link>
          <p>
            Ask a question in plain English and get SQL that runs, on 20 databases the models
            never saw in training.
          </p>
          <a className="icon-btn" href={REPO} aria-label="Spinneret on GitHub">
            <GitHubIcon />
          </a>
        </div>
        <nav aria-label="Product">
          <h2>Product</h2>
          <Link to="/query">Demo</Link>
          <Link to="/evaluation">Evaluation</Link>
          <Link to="/about">About the models</Link>
        </nav>
        <nav aria-label="Project">
          <h2>Project</h2>
          <a href={REPO}>Source code</a>
          <a href={`${REPO}#readme`}>Run it yourself</a>
          <a href={LICENSE_URL}>MIT licence</a>
        </nav>
        <nav aria-label="Trust">
          <h2>Trust</h2>
          <Link to="/privacy">Privacy and data</Link>
          <Link to="/privacy#queries">How queries run</Link>
        </nav>
      </div>
      <div className="footer-base">
        <span>© 2026 Spinneret. Released under the MIT licence.</span>
        <span>
          Built on the Spider dataset (Yu et al., 2018), CC BY-SA 4.0.
        </span>
      </div>
    </footer>
  );
}
