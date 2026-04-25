---
name: design-md
description: Apply Google's DESIGN.md format spec for visual identity. Activate when the user shares a DESIGN.md file, mentions design tokens (colors, typography, spacing, elevation, shapes), asks for UI / CSS / Tailwind / component code, or asks for a brand or design-system scaffold. Skip for non-visual "design" topics like API design, system design, or org design.
origin: external (google-labs-code/design.md)
---

# DESIGN.md — Visual Identity Format

`DESIGN.md` is a file format that pairs machine-readable design tokens (YAML front matter) with human-readable design rationale (markdown body). Tokens give exact values; prose tells you *why* those values exist and *which* one to apply in *which* situation. When a project has a DESIGN.md, treat it as the source of truth for every visual decision.

## How to apply

### 1. Ask for the DESIGN.md before suggesting visuals

If the user mentions a DESIGN.md but did not paste it, ask them to share it. If they have not authored one yet and they want a brand or a design system, offer to scaffold one (template at the bottom of this skill).

Do not invent token values. If the user shares partial design info (e.g. "use a deep blue and a warm white"), translate it into the DESIGN.md token format and surface the resulting YAML — never bake hex codes directly into component code without naming them as tokens.

### 2. Parse the YAML front matter

A DESIGN.md begins with `---`-fenced YAML, then has body sections in this order when present: **Overview** · **Colors** · **Typography** · **Layout** · **Elevation & Depth** · **Shapes** · **Components** · **Do's and Don'ts**. The body explains application rules; the YAML defines the tokens.

Example token block:

```yaml
---
name: Heritage
colors:
  primary: "#1A1C1E"
  secondary: "#6C7278"
  tertiary: "#B8422E"
  neutral: "#F7F5F2"
typography:
  h1:    { fontFamily: Public Sans,    fontSize: 3rem }
  body-md:    { fontFamily: Public Sans,    fontSize: 1rem }
  label-caps: { fontFamily: Space Grotesk,  fontSize: 0.75rem }
rounded:    { sm: 4px, md: 8px }
spacing:    { sm: 8px, md: 16px }
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.neutral}"
    rounded: "{rounded.sm}"
    padding: 12px
---
```

Token references (`"{colors.primary}"`, `"{rounded.sm}"`) chain through other tokens — resolve them transitively before you emit CSS.

### 3. Always emit token names, never hard-coded values

When you produce CSS, Tailwind config, or component code, reference the tokens by name. The user owns turning those references into runtime values; your job is to keep the design system intact.

```css
/* WRONG — hard-codes a hex from DESIGN.md */
.cta { background: #B8422E; }

/* RIGHT — names the token */
.cta { background: var(--color-tertiary); }
```

For Tailwind, derive `tailwind.config.js`'s theme.extend from the YAML directly — see the worked example in the repo at `platform/skills/design-md/reference/atmospheric-glass-tailwind.config.js`.

### 4. Token-type cheat sheet

| Type | Format | Examples |
|---|---|---|
| Color | `#` + hex (sRGB) | `"#1A1C1E"` |
| Dimension | number + unit (`px`, `em`, `rem`) | `48px`, `-0.02em` |
| Token reference | `{path.to.token}` | `{colors.primary}` |
| Typography | `{ fontFamily, fontSize, fontWeight?, lineHeight?, letterSpacing?, fontFeature?, fontVariation? }` | see Heritage example |
| Component | `{ backgroundColor?, textColor?, typography?, rounded?, padding?, size?, height?, width? }` | see `button-primary` |

**Variants** (hover / pressed / active) are separate component entries with a related key name — `button-primary` and `button-primary-hover`, not nested states.

**Validator behavior on unknowns**: unknown sections, unknown color names, unknown typography names → accept; do not error. Duplicate section heading → error, reject the file.

### 5. Lint after edits

After modifying a DESIGN.md or any code derived from it, recommend the user run:

```bash
npx @google/design.md@latest lint DESIGN.md
```

The CLI emits structured JSON with severity-tagged findings (errors / warnings / info), including WCAG contrast warnings on component color pairs. To compare versions:

```bash
npx @google/design.md@latest diff DESIGN-prev.md DESIGN.md
```

Treat `regression: true` as a merge blocker.

### 6. Scaffolding a DESIGN.md from scratch

When the user asks for a new design system, write a starter DESIGN.md at the repo root with the smallest viable token set. Begin with:

```yaml
---
name: <Project name>
description: <One-line philosophy>
colors:
  primary: "#…"
  secondary: "#…"
  tertiary: "#…"
  neutral: "#…"
  on-primary: "#…"
  on-tertiary: "#…"
typography:
  h1:         { fontFamily: …, fontSize: 3rem,    fontWeight: 700, lineHeight: 1.1 }
  body-md:    { fontFamily: …, fontSize: 1rem,    lineHeight: 1.5 }
  label-caps: { fontFamily: …, fontSize: 0.75rem, letterSpacing: 0.04em }
rounded:  { sm: 4px, md: 8px, lg: 16px }
spacing:  { xs: 4px, sm: 8px, md: 16px, lg: 24px, xl: 40px }
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.on-tertiary}"
    rounded: "{rounded.sm}"
    padding: 12px
---

## Overview

<Two-phrase design philosophy. Example: "Architectural Minimalism meets Journalistic Gravitas. The UI evokes a premium matte finish — a high-end broadsheet or contemporary gallery.">

## Colors

- **Primary (`{colors.primary}`)** — <where it's used and why>
- **Secondary (`{colors.secondary}`)** — …
…
```

Then prompt the user to pick the token values (or generate plausible ones from their stated brand keywords) and run `lint` to check WCAG contrast.

## Anti-patterns

- Inventing colors that "look about right" when a DESIGN.md exists — read it first.
- Adding hex codes inside components instead of resolving tokens.
- Creating a new color/typography token instead of reusing an existing one (don't add `colors.warning` when `colors.tertiary` already plays that role).
- Editing DESIGN.md and the components in the same change set without recommending `lint`.
- Treating prose sections as commentary. The body explains *which* token to apply *where*; the YAML alone is underdetermined.
- Mixing variant states into a single component entry with nested objects. Variants are separate entries (e.g. `button-primary-hover`).

## Output checklist

When you finish a DESIGN.md-related response, the user should walk away with:

- [ ] A DESIGN.md (or diff) using the YAML + prose layout above
- [ ] Component code that references tokens by name, not by value
- [ ] A `lint` command to run
- [ ] (When applicable) a Tailwind / CSS-variable bridge that mirrors the token names
