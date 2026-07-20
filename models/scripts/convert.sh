#!/usr/bin/env bash

for i in */models_eynollah/*;do
	rslv=$(readlink $i|sed 's,models_eynollah,reloaded/models_eynollah,');
	ln -srf $rslv $i;
done
