#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

set -e

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

ROOT="data"
DUMP_DIR="${ROOT}/wiki_dumps"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"

echo "Saving data in ""$DUMP_DIR"
read -r -p "Choose a language (e.g. en, bh, fr, etc.): " choice
LANG="$choice"
echo "Chosen language: ""$LANG"
DUMP_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_PATH="${DUMP_DIR}/${DUMP_FILE}"

if [ ! -f "${DUMP_PATH}" ]; then
	read -r -p "Continue to download (WARNING: This might be big and can take a long time!)(y/n)? " choice
	case "$choice" in
  	y|Y ) echo "Starting download...";;
  	n|N ) echo "Exiting";exit 1;;
  	* ) echo "Invalid answer";exit 1;;
	esac
	wget -c "https://dumps.wikimedia.org/""${LANG}""wiki/latest/""${DUMP_FILE}""" -P "${DUMP_DIR}"
else
  echo "${DUMP_PATH} already exists. Skipping download."
fi

DECOMP_FILE="${LANG}wiki-latest-pages-articles.xml"
if [ ! -f "${DUMP_DIR}/${DECOMP_FILE}" ]; then
	read -r -p "Continue to decompress wikipedia dump file (y/n)? " choice
	case "$choice" in
  	y|Y ) echo "Starting decompress...";;
  	n|N ) echo "Exiting";exit 1;;
  	* ) echo "Invalid answer";exit 1;;
	esac
	bzip2 -dk "${DUMP_PATH}"
else
	echo "${DUMP_DIR}/${DECOMP_FILE} already exists. Skipping decompression."