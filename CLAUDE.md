# CLAUDE.md - Ian's Standard Environment

## Core Identity
- **User:** Ian Bergman
- **Environment:** Hybrid (WSL2 Local + Replit Cloud)
- **Repo Host:** GitHub

## Universal Workflow
1. **Sync First:** Before starting major work, ensure `git pull` is run to catch Replit changes.
2. **Replit Compat:** Maintain `.replit` and `replit.nix`.
3. **Port:** Default to 5000 (Express) or 5173 (Vite).
4. **Commits:** Lowercase, descriptive, present tense (e.g., "fix login bug").

## Tech Stack Defaults (Node/Web)
- **Frontend:** React + Vite + Tailwind CSS
- **Backend:** Express or Next.js
- **Pkg Manager:** npm
- **New Code:** Use TypeScript and functional components.

## Commands
- **Start:** `npm run dev`
- **Install:** `npm install`
