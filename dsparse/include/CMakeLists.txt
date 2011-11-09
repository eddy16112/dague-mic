include(RulesPrecisions)
# reset variables
set(generated_headers "")

### generate the dsparse headers for all possible precisions
precisions_rules(generated_headers
                 "dsparse_z.h"
                 PRECISIONS "+;z;c;d;s")

add_custom_target(dsparse_includes ALL DEPENDS dsparse.h SOURCES
    ${generated_headers} )

### install the dsparse headers
install(FILES 
  dsparse.h 
  ${generated_headers}
  DESTINATION include)
foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include)
endforeach()

