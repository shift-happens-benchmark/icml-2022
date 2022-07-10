#!/bin/bash
# Check the documentation coverage of a list of PRs.

template_dir=$(dirname $0)

check_docstrings="interrogate -v \
--ignore-nested-functions \
--ignore-private \
--ignore-semiprivate \
--ignore-init-method"

mkdir -p _reports

generate_report() {
    pr=$1
    cat $template_dir/prefix.txt
    echo
    git diff --name-only main | grep .py$ | xargs $check_docstrings | sed -e 's/^-\+ \(RESULT[^\-]*\) -*/\1/g' | grep -v Summary | grep -v "Coverage for"
    echo
    cat $template_dir/suffix.txt
}

for pr in $(gh pr list --label task | cut -f1); do

    echo "Testing PR $pr"
    reportfile=_reports/${pr}.report
    gh pr checkout $pr
    generate_report $pr > $reportfile 
    # NOTE(stes) uncomment to post comments to PR.
    #gh pr comment $pr --body-file $reportfile

done
