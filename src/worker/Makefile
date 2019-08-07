

# git+git://github.com/andersx/qml@gradient_kernel --install-option="--compiler=intelem --fcompiler=intelem"

ENV=env
CONDA=conda
PYTHON=${ENV}/bin/python
PIP=${ENV}/bin/pip

QMLDIR=qmldevelop

env: conda_setup pip_setup

conda_setup:
	${CONDA} env create -f environment.yml -p ${ENV}

${QMLDIR}:
	git clone --depth 1 --single-branch --branch gradient_kernel https://github.com/andersx/qml ${QMLDIR}

build_qml: ${QMLDIR}
	cd ${QMLDIR}; ../${PYTHON} setup.py build --compiler=intelem --fcompiler=intelem

link_qml:
	cp -r ${QMLDIR}/build/lib.linux-x86_64-3.6/qml qml

pip_setup:
	${PIP} install -r requirements.txt

chemhelp:
	git clone https://github.com/charnley/chemhelp


# USAGE

status:
	${PYTHON} communication/redis_status.py --redis-task hello



# FCHL18

qml_fchl_path=qmldevelop/qml/fchl

build:
	mkdir -p build
#
# qmldevelop:
	# git clone --depth 1 --single-branch --branch develop git@github.com:qmlcode/qml.git qmldevelop
	# git clone --depth 1 --single-branch -branch gradient_kernel git@github.com/andersx/qml.git qmldevelop

fworkkernel:
	f2py -c qml_fchl_kernels.f90  ${qml_fchl_path}/ffchl_module.f90 ${qml_fchl_path}/ffchl_kernels.f90 -m fworkkernel

clean:
	rm -r qmldevelop
	rm *.so

clean-qml:
	${PIP} uninstall qml

