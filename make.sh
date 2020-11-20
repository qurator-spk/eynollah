all: build install

build:
       python3 setup.py build

install:
       python3 setup.py install --user
       cp sbb_newspapers_org_image/eynollah.py ~/bin/eynollah
       chmod +x ~/bin/eynollah
