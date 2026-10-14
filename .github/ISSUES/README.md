# JKL Missing Functionality - Issue Creation Guide

This directory contains detailed specifications for 10 GitHub issues that address missing functionality in the JKL texture compression library.

## Issue Files

Each `.md` file in this directory represents a complete GitHub issue:

1. **issue-01-fix-failing-jackal-test.md** - Critical bug fix
2. **issue-02-implement-cli.md** - CLI implementation
3. **issue-03-add-readme.md** - Documentation
4. **issue-04-implement-bc6-bc7.md** - Additional BC formats
5. **issue-05-anyblock-implementations.md** - Jackal support for all formats
6. **issue-06-texture-packing.md** - Texture atlas creation
7. **issue-07-mipmap-generation.md** - Mipmap support
8. **issue-08-encoder-api.md** - High-level API
9. **issue-09-comprehensive-tests.md** - Test coverage
10. **issue-10-examples-documentation.md** - Examples and docs

## How to Create Issues

### Automated Method (Recommended)

If you have `gh` CLI installed and configured:

```bash
# Create all issues at once
for file in .github/ISSUES/issue-*.md; do
  gh issue create --title "$(grep '^title:' "$file" | cut -d'"' -f2)" \
                  --body "$(sed '1,/^---$/d' "$file" | sed '1,/^---$/d')" \
                  --label "$(grep '^labels:' "$file" | sed 's/labels: //' | tr -d '[]"')"
done
```

### Manual Method

For each issue file:

1. Go to https://github.com/zakarumych/jkl/issues/new
2. Copy the title from the file's frontmatter
3. Copy the body content (everything after the second `---`)
4. Add the labels listed in the frontmatter
5. Submit the issue
6. Note the issue number
7. Update dependency references in other issues

### Issue Linking

After creating all issues, update dependency references:

- Issue #1 blocks: #2, #4, #5
- Issue #2 blocks: #3, #6
- Issue #1, #5 block: #7
- Issues #4, #5, #6, #7 block: #8
- All implementation issues block: #9
- Issues #2, #3, #8 block: #10

Use GitHub's issue reference syntax: `#N` or `Depends on #N`

## Priority Order

Issues are labeled by priority:

### Critical (Start Here)
- **Issue #1**: Fix failing Jackal test

### High Priority (Next)
- **Issue #2**: Implement CLI
- **Issue #3**: Add README
- **Issue #5**: AnyBlock implementations

### Medium Priority
- **Issue #4**: BC6/BC7 support
- **Issue #6**: Texture packing
- **Issue #7**: Mipmap generation
- **Issue #9**: Test coverage
- **Issue #10**: Examples

### Low Priority (Polish)
- **Issue #8**: High-level Encoder API

## Dependency Diagram

```
                    ┌─────────────┐
                    │  Issue #1   │
                    │  Fix Test   │
                    │ (CRITICAL)  │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  ┌──────────┐      ┌──────────┐     ┌──────────┐
  │Issue #2  │      │Issue #4  │     │Issue #5  │
  │   CLI    │      │ BC6/BC7  │     │ AnyBlock │
  │  (HIGH)  │      │ (MEDIUM) │     │  (HIGH)  │
  └────┬─────┘      └────┬─────┘     └────┬─────┘
       │                 │                 │
       ├────────┬────────┘                 │
       │        │                          │
       ▼        ▼                          │
  ┌──────────┐ ┌──────────┐               │
  │Issue #3  │ │Issue #6  │               │
  │  README  │ │  Packing │               │
  │  (HIGH)  │ │ (MEDIUM) │               │
  └──────────┘ └────┬─────┘               │
                    │                     │
                    │    ┌────────────────┘
                    │    │
                    ▼    ▼
               ┌──────────────┐
               │   Issue #7   │
               │   Mipmaps    │
               │   (MEDIUM)   │
               └──────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │Issue #8  │ │Issue #9  │ │Issue #10 │
  │ Encoder  │ │  Tests   │ │ Examples │
  │  (LOW)   │ │ (MEDIUM) │ │ (MEDIUM) │
  └──────────┘ └──────────┘ └──────────┘
```

## Labels Reference

Each issue should have these labels:

### Priority Labels
- `priority:critical` - Must fix immediately (#1)
- `priority:high` - Important (#2, #3, #5)
- `priority:medium` - Should have (#4, #6, #7, #9, #10)
- `priority:low` - Nice to have (#8)

### Type Labels
- `type:bug` - Bug fix (#1)
- `type:feature` - New feature (#2, #4, #5, #6, #7, #8)
- `type:documentation` - Documentation (#3, #10)
- `type:testing` - Testing (#9)

### Component Labels
- `component:cli` - CLI related (#2)
- `component:compression` - Compression formats (#1, #4, #5, #7)
- `component:packing` - Texture packing (#6)
- `component:api` - Public API (#8)

## Milestones (Optional)

Consider creating milestones to group related work:

### Milestone 1: Core Functionality
- Issue #1 (Fix test)
- Issue #5 (AnyBlock)
- Issue #2 (CLI)
- Issue #3 (README)

**Goal**: Make the library usable for basic compression

### Milestone 2: Complete Feature Set
- Issue #4 (BC6/BC7)
- Issue #6 (Packing)
- Issue #7 (Mipmaps)

**Goal**: Add all planned features

### Milestone 3: Production Ready
- Issue #8 (Encoder API)
- Issue #9 (Tests)
- Issue #10 (Examples)

**Goal**: Polish for production use and public release

## Project Board (Optional)

Create a GitHub Project board with columns:
- **Backlog**: All issues
- **To Do**: Issues ready to work on
- **In Progress**: Currently being worked on
- **In Review**: Pending code review
- **Done**: Completed

## Notes for Issue Creation

1. **Consistent Formatting**: All issue files use a consistent format for easy parsing
2. **Frontmatter**: Contains metadata (title, labels, assignees)
3. **Detailed Content**: Each issue has comprehensive description, requirements, and acceptance criteria
4. **Examples**: Most issues include code examples
5. **Dependencies**: Clearly marked dependencies and blocking relationships

## Updating Issues

As work progresses, update issues with:
- Progress comments
- Links to PRs
- Questions or clarifications
- Test results
- Performance measurements

## Cross-References

Remember to link:
- Related issues (using `#N` syntax)
- PRs that address issues (using "Fixes #N" in PR description)
- Documentation updates
- Breaking changes

## Questions?

If you have questions about any issue:
1. Check the detailed issue file in this directory
2. Refer to the main ACTION_ITEMS.md file
3. Review the existing code and tests
4. Ask in issue comments for clarification
