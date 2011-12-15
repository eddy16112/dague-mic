subs = {
  'all' : [ ## Special key: Changes are applied to all applicable conversions automatically
    [None,None]
  ],
  'mixed' : [
    ['zc','ds'],
    ('ZC','DS'),
    ('zc','ds'),
    ('PLASMA_Complex64_t','double'),
    ('PLASMA_Complex32_t','float'),
    ('PlasmaComplexDouble','PlasmaRealDouble'),
    ('PlasmaComplexFloat','PlasmaRealFloat'),
    ('Dague_Complex64_t',   'double'           ),
    ('Dague_Complex32_t',   'float'            ),
    ('matrix_ComplexDouble','matrix_RealDouble'),
    ('matrix_ComplexFloat', 'matrix_RealFloat' ),
    ('zlange','dlange'),
    ('zlag2c','dlag2s'),
    ('clag2z','slag2d'),
    ('zlacpy','dlacpy'),
    ('zgemm','dgemm'),
    ('zherk','dsyrk'),
    ('zlansy','dlansy'),
    ('zaxpy','daxpy'),
    ('pzgetrf','pdgetrf'),
    ('pcgetrf','psgetrf'),
    ('ztrsm','dtrsm'),
    ('ctrsm','strsm'),
    ('CBLAS_SADDR',''),
    ('zlarnv','dlarnv'),
    ('zgesv','dgesv'),
    ('zhemm','dsymm'),
    ('zlanhe','dlansy'),
    ('zlaghe','dlagsy'),
    ('ztrmm','dtrmm'),
    ('ctrmm','strmm'),
    ('zherbt','dsyrbt'),
    ('cherbt','ssyrbt'),
    ('zhbrdt','dsbrdt'),
    ('chbrdt','ssbrdt'),
    ('Conj',''),
    ('zpotrf','dpotrf'),
    ('cpotrf','spotrf'),
    ('PLASMA_Alloc_Workspace_zgels','PLASMA_Alloc_Workspace_dgels'),
    ('pcgeqrf','psgeqrf'),
    ('pcunmqr','psormqr'),
    ('pcgelqf','psgelqf'),
    ('pcunmlq','psormlq'),
    ('pzgeqrf','pdgeqrf'),
    ('pzunmqr','pdormqr'),
    ('pzgelqf','pdgelqf'),
    ('pzherbt','pdsyrbt'),
    ('pcherbt','pssyrbt'),
    ('pzhbrdt','pdsbrdt'),
    ('pchbrdt','pssbrdt'),
    ('pzunmlq','pdormlq'),
    ('plasma_pclapack','plasma_pslapack'),
    ('plasma_pzlapack','plasma_pdlapack'),
    ('plasma_pctile','plasma_pstile'),
    ('plasma_pztile','plasma_pdtile'),
  ],
  'normal' : [ ## Dictionary is keyed on substitution type
    ['s','d','c','z'], ## Special Line Indicating type columns

    ('#define PRECISION_s', '#define PRECISION_d', '#define PRECISION_c', '#define PRECISION_z' ),
    ('#define REAL',        '#define REAL',        '#define COMPLEX',     '#define COMPLEX'     ),
    ('#undef COMPLEX',      '#undef COMPLEX',      '#undef REAL',         '#undef REAL'         ),
    ('#define SINGLE',      '#define DOUBLE',      '#define SINGLE',      '#define DOUBLE'      ),
    ('#undef DOUBLE',       '#undef SINGLE',       '#undef DOUBLE',       '#undef SINGLE'       ),
    ('float',               'double',              'Dague_Complex32_t',   'Dague_Complex64_t'   ),
    ('RealFloat',    'RealDouble',   'ComplexFloat', 'ComplexDouble'),
    ('float',               'double',              'float',               'double'              ),
    ('RealFloat',    'RealDouble',   'RealFloat',    'RealDouble'   ),
    ('MPI_FLOAT',           'MPI_DOUBLE',          'MPI_COMPLEX',         'MPI_DOUBLE_COMPLEX'  ),
    ('smatrix',             'dmatrix',             'cmatrix',             'zmatrix'             ),
    ('stwoDBC',             'dtwoDBC',             'ctwoDBC',             'ztwoDBC'             ),
    ('cimagf',              'cimag',               'cimagf',              'cimag'               ),
    ('cblas_sasum','cblas_dasum','cblas_scasum','cblas_dzasum'),
    ('CORE_sasum','CORE_dasum','CORE_scasum','CORE_dzasum'),
    ('core_sasum','core_dasum','core_scasum','core_dzasum'),
    ('sread',    'dread',    'cread',    'zread'   ),
    ('s_read',   'd_read',   'c_read',   'z_read'  ),
    ('s_pastix', 'd_pastix', 'c_pastix', 'z_pastix'),
    ('S_pastix', 'D_pastix', 'C_pastix', 'Z_pastix'),
    ('S_Csc',    'D_Csc',    'C_Csc',    'Z_Csc'   ),
    ('S_Build',  'D_Build',  'C_Build',  'Z_Build' ),
    ('strdv', 'dtrdv', 'ctrdv', 'ztrdv'),
    ('sytra1','sytra1','hetra1','hetra1'),
    ('ssygst','dsygst','chegst','zhegst'),
    ('SSYGST','DSYGST','CHEGST','ZHEGST'),
    ('ssterf','dsterf','ssterf','dsterf'),
    ('ssytrd','dsytrd','chetrd','zhetrd'),
    ('SSYTRD','DSYTRD','CHETRD','ZHETRD'),
    ('STILE','DTILE','CTILE','ZTILE'),
    ('stile','dtile','ctile','ztile'),
    ('slag2d','dlag2s','clag2z','zlag2c'),
    ('ssyrfb','dsyrfb','cherfb','zherfb'),
    ('ssyrf','dsyrf','cherf','zherf'),
    ('saxpy','daxpy','caxpy','zaxpy'),
    ('ssymm','dsymm','chemm','zhemm'),
    ('SSYMM','DSYMM','CHEMM','ZHEMM'),
    ('ssyr','dsyr','cher','zher'),
    ('SSYR','DSYR','CHER','ZHER'),
    ('SYR','SYR','HER','HER'),
    ('sgesv','dgesv','cgesv','zgesv'),
    ('SUNGESV','SUNGESV','CUNGESV','CUNGESV'),
    ('SGESV','DGESV','CGESV','ZGESV'),
    ('SGESV','SGESV','CGESV','CGESV'),
    ('SPOSV','SPOSV','CPOSV','CPOSV'),
    ('sgels','dgels','cgels','zgels'),
    ('SGELS','DGELS','CGELS','ZGELS'),
    ('sgemm','dgemm','cgemm','zgemm'),
    ('SGEMM','DGEMM','CGEMM','ZGEMM'),
    ('sposv','dposv','cposv','zposv'),
    ('SPOSV','DPOSV','CPOSV','ZPOSV'),
    ('ssymm','dsymm','csymm','zsymm'),
    ('SSYMM','DSYMM','CSYMM','ZSYMM'),
    ('ssyrk','dsyrk','csyrk','zsyrk'),
    ('SSYRK','DSYRK','CSYRK','ZSYRK'),
    ('strmm','dtrmm','ctrmm','ztrmm'),
    ('STRMM','DTRMM','CTRMM','ZTRMM'),
    ('ssytrf','dsytrf','chetrf','zhetrf'),
    ('ssytrf','dsytrf','csytrf','zsytrf'),
    ('ssyrbt','dsyrbt','cherbt','zherbt'),
    ('SSYRBT','DSYRBT','CHERBT','ZHERBT'),
    ('ssbrdt','dsbrdt','chbrdt','zhbrdt'),
    ('SSBRDT','DSBRDT','CHBRDT','ZHBRDT'),
    ('strsm','dtrsm','ctrsm','ztrsm'),
    ('STRSM','DTRSM','CTRSM','ZTRSM'),
    ('sgelq2','dgelq2','cgelq2','zgelq2'),
    ('sgelqf','dgelqf','cgelqf','zgelqf'),
    ('SGELQF','DGELQF','CGELQF','ZGELQF'),
    ('sgelqs','dgelqs','cgelqs','zgelqs'),
    ('SGELQS','DGELQS','CGELQS','ZGELQS'),
    ('sgeqr2','dgeqr2','cgeqr2','zgeqr2'),
    ('sgeqrf','dgeqrf','cgeqrf','zgeqrf'),
    ('SGEQRF','DGEQRF','CGEQRF','ZGEQRF'),
    ('sgeqrs','dgeqrs','cgeqrs','zgeqrs'),
    ('SGEQRS','DGEQRS','CGEQRS','ZGEQRS'),
    ('sgetf2','dgetf2','cgetf2','zgetf2'),
    ('sgetrf','dgetrf','cgetrf','zgetrf'),
    ('SGETRF','DGETRF','CGETRF','ZGETRF'),
    ('sgetrs','dgetrs','cgetrs','zgetrs'),
    ('SGETRS','DGETRS','CGETRS','ZGETRS'),
    ('sgerbb','dgerbb','cgerbb','zgerbb'),
    ('SGERBB','DGERBB','CGERBB','ZGERBB'),
    ('splrnt','dplrnt','cplrnt','zplrnt'),
    ('splgsy','dplgsy','cplgsy','zplgsy'),
    ('splgsy','dplgsy','cplghe','zplghe'),
    ('slaset','dlaset','claset','zlaset'),
    ('slacgv','dlacgv','clacgv','zlacgv'),
    ('slacpy','dlacpy','clacpy','zlacpy'),
    ('slagsy','dlagsy','claghe','zlaghe'),
    ('slagsy','dlagsy','clagsy','zlagsy'),
    ('SLANGE','DLANGE','CLANGE','ZLANGE'),
    ('SLANSY','DLANSY','CLANHE','ZLANHE'),
    ('SLANSY','DLANSY','CLANSY','ZLANSY'),
    ('SLANTR','DLANTR','CLANTR','ZLANTR'),
    ('slange','dlange','clange','zlange'),
    ('slansy','dlansy','clanhe','zlanhe'),
    ('slansy','dlansy','clansy','zlansy'),
    ('slantr','dlantr','clantr','zlantr'),
    ('slarfb','dlarfb','clarfb','zlarfb'),
    ('slarfg','dlarfg','clarfg','zlarfg'),
    ('slarft','dlarft','clarft','zlarft'),
    ('slarnv','dlarnv','clarnv','zlarnv'),
    ('slaswp','dlaswp','claswp','zlaswp'),
    ('spotrf','dpotrf','cpotrf','zpotrf'),
    ('spotrf','dpotrf','cpotrf','zpotrf'),
    ('SPOTRF','DPOTRF','CPOTRF','ZPOTRF'),
    ('spotrs','dpotrs','cpotrs','zpotrs'),
    ('SPOTRS','DPOTRS','CPOTRS','ZPOTRS'),
    ('sorglq','dorglq','cunglq','zunglq'),
    ('SORGLQ','DORGLQ','CUNGLQ','ZUNGLQ'),
    ('sorgqr','dorgqr','cungqr','zungqr'),
    ('SORGQR','DORGQR','CUNGQR','ZUNGQR'),
    ('sormlq','dormlq','cunmlq','zunmlq'),
    ('SORMLQ','DORMLQ','CUNMLQ','ZUNMLQ'),
    ('sormqr','dormqr','cunmqr','zunmqr'),
    ('SORMQR','DORMQR','CUNMQR','ZUNMQR'),
    ('ORMQR','ORMQR','UNMQR','UNMQR'),
    ('slamch','dlamch','slamch','dlamch'),
    ('slarnv','dlarnv','slarnv','dlarnv'),
    ('slauum','dlauum','clauum','zlauum'),
    ('spotri','dpotri','cpotri','zpotri'),
    ('strtri','dtrtri','ctrtri','ztrtri'),
    ('strsmpl','dtrsmpl','ctrsmpl','ztrsmpl'),
    ('STRSMPL','DTRSMPL','CTRSMPL','ZTRSMPL'),
    ('scsc2cblk','dcsc2cblk','ccsc2cblk','zcsc2cblk'),
    ('ger','ger','gerc','gerc'),
    ('ger','ger','geru','geru'),
    ('symm','symm','hemm','hemm'),
    ('syr','syr','her','her'),
    ('syrbt','syrbt','herbt','herbt'),
    ('ssyev','dsyev','cheev','zheev'),
    ('syrfb1','syrfb1','herfb1','herfb1'),
    ('lansy','lansy','lanhe','lanhe'),
    ('\*\*T','\*\*T','\*\*H','\*\*H'),
    ('BLAS_s','BLAS_d','BLAS_s','BLAS_d'),
    ('BLAS_s','BLAS_d','BLAS_c','BLAS_z'),
    ('cblas_is','cblas_id','cblas_ic','cblas_iz'),
    ('cblas_s','cblas_d','cblas_c','cblas_z'),
    ('','','CBLAS_SADDR','CBLAS_SADDR'),
    ('CblasTrans','CblasTrans','CblasConjTrans','CblasConjTrans'),
    ('REAL','DOUBLE_PRECISION','COMPLEX','COMPLEX_16'),
    ('','','conjf','conj'),
    ('fabsf','fabs','fabsf','fabs'),
    ('fabsf','fabs','cabsf','cabs'),
    ('sqrtf','sqrt','csqrtf','csqrt'),
    ('float','double','float _Complex','double _Complex'),
    ('float','double','float','double'),
    ('lapack_slamch','lapack_dlamch','lapack_slamch','lapack_dlamch'),
    ('float','double','PLASMA_Complex32_t','PLASMA_Complex64_t'),
    ('float','double','PLASMA_voidComplex32_t','PLASMA_voidComplex64_t'),
    ('PLASMA_sor','PLASMA_dor','PLASMA_cun','PLASMA_zun'),
    ('PlasmaRealFloat','PlasmaRealDouble','PlasmaComplexFloat','PlasmaComplexDouble'),
    ('PlasmaTrans','PlasmaTrans','PlasmaConjTrans','PlasmaConjTrans'),
    ('stesting','dtesting','ctesting','ztesting'),
    ('SAUXILIARY','DAUXILIARY','CAUXILIARY','ZAUXILIARY'),
    ('sauxiliary','dauxiliary','cauxiliary','zauxiliary'),
    ('scheck','dcheck','ccheck','zcheck'),
    ('stile','dtile','ctile','ztile'),
    ('control_s',   'control_d',   'control_c',   'control_z'  ),
    ('compute_s',   'compute_d',   'compute_c',   'compute_z'  ),
    ('CORE_S',      'CORE_D',      'CORE_C',      'CORE_Z'     ),
    ('CORE_s',      'CORE_d',      'CORE_c',      'CORE_z'     ),
    ('CORE_s',      'CORE_d',      'CORE_s',      'CORE_d'     ),
    ('core_s',      'core_d',      'core_c',      'core_z'     ),
    ('coreblas_s',  'coreblas_d',  'coreblas_c',  'coreblas_z' ),
    ('example_s',   'example_d',   'example_c',   'example_z'  ),
    ('lapack_s',    'lapack_d',    'lapack_c',    'lapack_z'   ),
    ('PLASMA_s',    'PLASMA_d',    'PLASMA_c',    'PLASMA_z'   ),
    ('PLASMA_S',    'PLASMA_D',    'PLASMA_C',    'PLASMA_Z'   ),
    ('plasma_s',    'plasma_d',    'plasma_c',    'plasma_z'   ),
    ('plasma_ps',   'plasma_pd',   'plasma_pc',   'plasma_pz'  ),
    ('dplasma_s',   'dplasma_d',   'dplasma_c',   'dplasma_z'  ),
    ('dsparse_s',   'dsparse_d',   'dsparse_c',   'dsparse_z'  ),
    ('DSPARSE_S',   'DSPARSE_D',   'DSPARSE_C',   'DSPARSE_Z'  ),
    ('matrix_s',    'matrix_d',    'matrix_c',    'matrix_z'   ),
    ('testing_ds',  'testing_ds',  'testing_zc',  'testing_zc' ),
    ('testing_s',   'testing_d',   'testing_c',   'testing_z'  ),
    ('TESTING_S',   'TESTING_D',   'TESTING_C',   'TESTING_Z'  ),
    ('twoDBC_s',    'twoDBC_d',    'twoDBC_c',    'twoDBC_z'   ),
    ('workspace_s', 'workspace_d', 'workspace_c', 'workspace_z'),
  ],
  'tracing' : [
    ['plain','tau'],
    ('(\w+\*?)\s+(\w+)\s*\(([a-z* ,A-Z_0-9]*)\)\s*{\s+(.*)\s*#pragma tracing_start\s+(.*)\s+#pragma tracing_end\s+(.*)\s+}',r'\1 \2(\3){\n\4tau("\2");\5tau();\6}'),
    ('\.c','.c.tau'),
  ],
};
