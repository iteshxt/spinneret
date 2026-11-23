import { Link } from "react-router-dom";
import { PageHead, Reveal } from "../components/Motion";
import { REPO } from "../site";

const SECTIONS = [
  ["send", "What you send, and where it goes"],
  ["server", "What the server keeps"],
  ["browser", "What stays in your browser"],
  ["queries", "How queries run"],
  ["others", "Other sites"],
  ["hosted", "If someone else hosts Spinneret"],
  ["answers", "Checking the answers"],
  ["contact", "Questions and changes"],
] as const;

export function Privacy() {
  return (
    <div className="page doc">
      <PageHead title="Privacy and data">
        <p>
          Spinneret has no accounts, no cookies, no analytics and no ads. Your questions go to the
          Spinneret server only so it can answer them, and the server does not keep a copy.
        </p>
        <p className="muted">Last updated 23 November 2026.</p>
      </PageHead>

      <div className="doc-layout">
        <nav className="doc-toc" aria-label="On this page">
          {SECTIONS.map(([id, title]) => (
            <a key={id} href={`#${id}`}>
              {title}
            </a>
          ))}
        </nav>

        <div className="doc-body">
          <Reveal as="section">
            <h2 id="send">What you send, and where it goes</h2>
            <p>When you use the demo, your browser sends the Spinneret server:</p>
            <ul>
              <li>the question you type, the database you picked and the model you chose;</li>
              <li>any SQL you edit and run yourself.</li>
            </ul>
            <p>
              Nothing else is attached: no name, no email, no account and no tracking identifier.
              If you run Spinneret on your own computer, as the{" "}
              <a href={`${REPO}#readme`}>README</a> describes, the server is your own machine and
              nothing leaves it.
            </p>
          </Reveal>

          <Reveal as="section">
            <h2 id="server">What the server keeps</h2>
            <p>
              Nothing about you. Questions, generated SQL and results are held in memory only while
              your request is being answered, and are never written to disk, to a database or to a
              log file.
            </p>
            <p>
              The web server prints a one-line access record for each request to its console, for
              example <code>POST /api/query 200</code> with the address the request came from. That
              line does not include your question or your SQL, and it is not saved anywhere unless
              the person running the server chooses to keep the console output.
            </p>
          </Reveal>

          <Reveal as="section">
            <h2 id="browser">What stays in your browser</h2>
            <ul>
              <li>
                <strong>Session history.</strong> Your last 20 questions are kept in this tab's
                session storage so you can go back to them. They disappear when you close the tab.
              </li>
              <li>
                <strong>Theme.</strong> If you switch between light and dark, that choice is kept in
                local storage on this device.
              </li>
            </ul>
            <p>
              Neither is sent anywhere. You can remove both at any time by clearing this site's data
              in your browser settings.
            </p>
          </Reveal>

          <Reveal as="section">
            <h2 id="queries">How queries run</h2>
            <p>
              The 20 databases in the demo are the public sample databases from the Spider dataset
              (about singers, pets, flights, cars and so on). They contain no data about you, and
              the demo cannot connect to any other database.
            </p>
            <p>Every query, whether the model wrote it or you edited it, is checked and limited:</p>
            <ul>
              <li>only a single <code>SELECT</code> or <code>WITH</code> statement can run;</li>
              <li>
                statements that change data or settings, such as <code>INSERT</code>,{" "}
                <code>UPDATE</code>, <code>DELETE</code>, <code>DROP</code>, <code>ATTACH</code> or{" "}
                <code>PRAGMA</code>, are refused before they reach the database;
              </li>
              <li>
                databases are opened read-only, with SQLite's <code>query_only</code> mode on as
                well, so a write would fail even if a check missed it;
              </li>
              <li>a query is stopped after 5 seconds, and at most 500 rows are returned.</li>
            </ul>
          </Reveal>

          <Reveal as="section">
            <h2 id="others">Other sites</h2>
            <p>
              The page loads nothing from other sites: its fonts and code are served by the
              Spinneret server itself. Links to GitHub or Hugging Face open those sites, which have
              their own privacy policies.
            </p>
            <p>
              When the server starts, it downloads the model weights from Hugging Face once. That
              download comes from the server, not from your browser, and includes nothing you
              typed.
            </p>
          </Reveal>

          <Reveal as="section">
            <h2 id="hosted">If someone else hosts Spinneret</h2>
            <p>
              This page describes the software as it is published. Anyone running a copy can change
              it, for example to add their own logging. If you use a copy hosted by someone else,
              their own policy applies.
            </p>
          </Reveal>

          <Reveal as="section">
            <h2 id="answers">Checking the answers</h2>
            <p>
              The models are right about two times in three on databases they have not seen (see{" "}
              <Link to="/evaluation">Evaluation</Link>). The demo always shows the SQL it ran, so
              you can check it. Don't rely on an answer without reading the query.
            </p>
          </Reveal>

          <Reveal as="section">
            <h2 id="contact">Questions and changes</h2>
            <p>
              Questions or concerns can be raised as an issue on{" "}
              <a href={`${REPO}/issues`}>GitHub</a>. If anything on this page changes, the date at
              the top will change with it, and the history is in the repository.
            </p>
          </Reveal>
        </div>
      </div>
    </div>
  );
}
