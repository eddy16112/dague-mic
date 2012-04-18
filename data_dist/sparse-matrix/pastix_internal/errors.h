/*
  File: errors.h

  Defines and macros used for error management.

  Authors:
    Mathieu Faverge - faverge@labri.fr
    Xavier   LACOSTE - lacoste@labri.fr
 */
#ifndef PASTIX_ERRORS_H
#define PASTIX_ERRORS_H
/*
  Group: Defines
*/
/*
  define: DUMP_FILENAME

  File where to dump parameters array if an error occur.
  
*/
#define DUMP_FILENAME "./dump.txt"

/* 
   Group: Macros
*/

/*
  Macro: ASSERT
  
  Prints an error and calls <EXIT> if a value is false.

  Parameters:
    expr   - Value to test.
    module - Module in which the test is performed.
*/
#define ASSERT(expr,module){						\
    if((expr) == 0)							\
    {									\
      fprintf(stderr,"%s:%d error in assert (%s)\n",			\
	      __FILE__, __LINE__, #expr);				\
      EXIT((module),ASSERT_ERR);					\
    }									\
  }


/*
  Macro: ASSERTDBG
  
  If FLAG_ASSERT is defined, call <ASSERT>

  Parameters:
    expr   - Value to test.
    module - Module in which the test is performed.
*/
#ifdef FLAG_ASSERT
#define ASSERTDBG(expr,module) {ASSERT(expr,module)}
#else
#define ASSERTDBG(expr,module) {}
#endif

/*
  Macro: ASSERT_DEBUG
  
  Check expression and print error if the assertion is
  not verified.

  Only performed if PASTIX_DEBUG is defined and 
  *dbg_flag* is not null.

  Parameters:
    expr     - Expression to check. 
    dbg_flag - Flag indicating if check has
               to be performed.
*/
#ifdef PASTIX_DEBUG
#define ASSERT_DEBUG(expr, dbg_flag){					\
    if(dbg_flag == 1 && (expr) == 0){					\
      fprintf(stderr,"%s:%d error in assert (%s)\n",			\
	      __FILE__, __LINE__, #expr);				\
      abort();}}
#else
#define ASSERT_DEBUG(expr, dbg_flag)
#endif

#ifdef PASTIX_DEBUG_NAN
/*
 * Macro:  TAB_CHECK_NAN
 *
 * Check if any entry of the array is equal to NaN or infinity.
 *
 * Parameters:
 *   tab  - The array to check.
 *   size - The number of entries in the array.
 */
#define TAB_CHECK_NAN(tab, size)                \
  do {                                          \
    INT pdn_i;                                  \
    for (pdn_i = 0; pdn_i < (size); pdn_i++)    \
      {                                         \
        CHECK_NAN((tab)[i]);                    \
      }                                         \
  } while (0)

/*
 * Macro:  CHECK_NAN
 *
 * Check if the argument is equal to NaN or infinity.
 *
 * Parameters:
 *   expr - The value to check.
 */
#define CHECK_NAN(expr) {					\
    ASSERT_DEBUG(!isnan(expr), DBG_SOPALIN_NAN);		\
    ASSERT_DEBUG(!isinf(expr), DBG_SOPALIN_INF);		\
  }
#else
#define TAB_CHECK_NAN(tab, size)                \
  do {                                          \
  } while (0)
#define CHECK_NAN(expr)
#endif
/*
  Macro: EXIT
  
  Set IPARM_ERROR_NUMBER  to module+error, dumps parameters and exit.

  Parameters: 
    module - Module where the error occurs.
    error  - Value to set IPARM_ERROR_NUMBER to.
*/
#ifdef EXIT_ON_SIGSEGV
#define EXIT(module,error) { *(int *)0 = 42; }
#else
#define EXIT(module,error) { abort(); }
#endif

/*
  Macro: RETURN
  
  If value is different from NO_ERR, sets IPARM_ERROR_NUMBER to module+error,
  dumps parameters and return value.

  Parameters: 
    value  - Value to test.
    module - Module where the error occurs.
    error  - Value to set <IPARM_ERROR_NUMBER> to.
*/
#define RETURN(value,module,error) {		\
    return(value);				\
  }


/*
  Macro: RETURN_ERROR

  Return the value

  Parameters: 
    value  - Value to test.
*/
#define RETURN_ERROR(value) {				\
    return value ;					\
  }

/*
  Macro: MALLOC_ERROR
  
  Prints an error message and call <EXIT> with MOD_UNKNOWN as module and
  ALLOC_ERR as error.
*/
#define MALLOC_ERROR(x) {\
    errorPrint("%s allocation (line=%d,file=%s)\n",(x),__LINE__,__FILE__);\
    EXIT(MOD_UNKNOWN,ALLOC_ERR);}

#endif /* PASTIX_ERRORS_H*/
