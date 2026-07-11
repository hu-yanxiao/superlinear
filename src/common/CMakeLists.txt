set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/bfgs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stdafx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  )

target_sources(mlp PRIVATE
  ${SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/lbfgs.cpp
  )
target_sources(libinterface PRIVATE ${SOURCES})
