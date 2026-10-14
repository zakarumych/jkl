#!/bin/bash
# Script to create GitHub issues from markdown files
# Usage: ./create_issues.sh

set -e

ISSUE_DIR=".github/ISSUES"
REPO="zakarumych/jkl"

echo "=========================================="
echo "JKL Issue Creation Script"
echo "=========================================="
echo ""
echo "This script will create 10 GitHub issues for the JKL project."
echo "Repository: $REPO"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    echo ""
    echo "Or create issues manually using the markdown files in $ISSUE_DIR/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Please run: gh auth login"
    exit 1
fi

echo "GitHub CLI detected and authenticated ✓"
echo ""

# Function to extract title from frontmatter
extract_title() {
    grep '^title:' "$1" | sed 's/^title: *"\(.*\)"$/\1/'
}

# Function to extract labels from frontmatter
extract_labels() {
    grep '^labels:' "$1" | sed 's/^labels: *\[\(.*\)\]$/\1/' | tr -d '"' | tr ',' '\n' | sed 's/^ *//'
}

# Function to extract body (skip frontmatter)
extract_body() {
    sed '1,/^---$/d' "$1" | sed '1,/^---$/d'
}

# Confirm before proceeding
echo "About to create the following issues:"
echo ""

issue_num=1
for file in "$ISSUE_DIR"/issue-*.md; do
    if [ -f "$file" ]; then
        title=$(extract_title "$file")
        echo "$issue_num. $title"
        issue_num=$((issue_num + 1))
    fi
done

echo ""
read -p "Do you want to proceed? (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Creating issues..."
echo ""

# Array to store created issue numbers
declare -a issue_numbers

# Create issues
issue_num=1
for file in "$ISSUE_DIR"/issue-*.md; do
    if [ -f "$file" ]; then
        title=$(extract_title "$file")
        body=$(extract_body "$file")
        labels=$(extract_labels "$file")
        
        echo "Creating issue $issue_num: $title"
        
        # Convert labels array to comma-separated list for gh CLI
        label_args=""
        while IFS= read -r label; do
            if [ -n "$label" ]; then
                label_args="$label_args --label \"$label\""
            fi
        done <<< "$labels"
        
        # Create the issue and capture the URL
        issue_url=$(echo "$body" | gh issue create \
            --repo "$REPO" \
            --title "$title" \
            $label_args \
            --body-file - 2>&1 | grep -o 'https://github.com[^ ]*')
        
        # Extract issue number from URL
        issue_number=$(echo "$issue_url" | grep -o '[0-9]*$')
        issue_numbers+=("$issue_number")
        
        echo "  ✓ Created: $issue_url"
        echo ""
        
        issue_num=$((issue_num + 1))
        
        # Rate limiting: sleep between issues
        sleep 2
    fi
done

echo "=========================================="
echo "All issues created successfully!"
echo "=========================================="
echo ""
echo "Created issues:"
for i in "${!issue_numbers[@]}"; do
    num=$((i + 1))
    echo "  Issue $num: #${issue_numbers[$i]}"
done

echo ""
echo "Next steps:"
echo "1. Review the created issues at: https://github.com/$REPO/issues"
echo "2. Update cross-references between issues if needed"
echo "3. Create a project board to track progress"
echo "4. Assign issues to team members"
echo ""
echo "Note: You may need to manually update issue references like '#1', '#2', etc."
echo "      in the issue descriptions now that the actual issue numbers are known."
echo ""
