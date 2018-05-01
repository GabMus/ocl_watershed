TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp
QMAKE_CXXFLAGS += -std=c++0x
LIBS += -lOpenCL

DISTFILES += \
    LICENSE \
    ocl_source.cl \
    include/cxxopts_LICENSE

copyfiles.commands = cp $$PWD/*.cl $$OUT_PWD/

QMAKE_EXTRA_TARGETS += copyfiles
POST_TARGETDEPS += copyfiles

HEADERS += \
    cl_errorcheck.hpp \
    imagelib.hpp \
    io_helper.hpp \
    ocl_helper.hpp \
    graph.hpp \
    include/cxxopts.hpp
