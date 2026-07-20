#!/bin/bash
#
# Upload big files to Zenodo.
#
# usage: ./zenodo_upload.sh [deposition id] [filename] [--verbose|-v]
#
# Taken from https://github.com/jhpoelen/zenodo-upload
# 
# Copyright (c) 2019 Jorrit Poelen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

set -e

VERBOSE=0
if [ "$3" == "--verbose" ] || [ "$3" == "-v" ]; then
    VERBOSE=1
fi

# strip deposition url prefix if provided; see https://github.com/jhpoelen/zenodo-upload/issues/2#issuecomment-797657717
DEPOSITION=$( echo $1 | sed 's+^http[s]*://zenodo.org/deposit/++g' )
FILEPATH="$2"
FILENAME=$(echo $FILEPATH | sed 's+.*/++g')
FILENAME=${FILENAME// /%20}
ZENODO_ENDPOINT=${ZENODO_ENDPOINT:-https://zenodo.org}

DEPOSIT_URL=${ZENODO_ENDPOINT}/api/deposit/depositions/"$DEPOSITION"?access_token="$ZENODO_TOKEN"
BUCKET=$(curl $DEPOSIT_URL | jq --raw-output .links.bucket)

if [ "$VERBOSE" -eq 1 ]; then
    echo "Deposition ID: $DEPOSITION"
    echo "File path: $FILEPATH"
    echo "File name: $FILENAME"
    echo "Bucket URL: $BUCKET"
    echo "Upload URL: $BUCKET/$FILENAME"
    echo "Deposit URL: $DEPOSIT_URL"
    echo "Uploading file..."
fi

set -x
curl --progress-bar \
    --retry 5 \
    --retry-delay 5 \
    -o ${FILENAME}.upload.log \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    --upload-file "$FILEPATH" \
    "$BUCKET/$FILENAME"

# curl -i \
#      -H "Authorization: Bearer $ZENODO_TOKEN" \
#      -X POST \
#      -F name=$FILENAME \
#      -F file=@$FILEPATH \
# 	https://zenodo.org/api/deposit/depositions/$DEPOSITION/files
