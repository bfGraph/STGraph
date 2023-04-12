============================
Compiler Interface (zoom_v2)
============================

.. py:class:: Context

    GraphInfo is a named tuple data structure whose fields can be
    accessed using indexing and the following field names 

    ``GraphInfo(number_of_nodes, number_of_edges, in_row_offsets, in_col_indices, in_eids, out_row_offsets, out_col_indices, out_eids, nbits)``
 
    .. py:attribute:: _f
        :type: 
    .. py:attribute:: _nspace
        :type: 
    .. py:attribute:: _entry_count
        :type: int
    .. py:attribute:: _run_cb
        :type: 
    .. py:attribute:: _input_cache
        :type: dict
    .. py:attribute:: _graph_info_cache
        :type: 
    .. py:attribute:: _executor_cache
        :type: 

.. py:class:: CtxManager

    .. py:attribute:: _ctx_map
        :type: dict
    .. py:attribute:: _run_cb
        :type: 