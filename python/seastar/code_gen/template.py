from jinja2 import Template

from collections import namedtuple

from .compiler import compile_cuda

header_tpl = Template(
"""
{%for header in headers%}
{{'#include ' + header}}{%endfor%}
"""
)

tpl_v2 = Template(
"""
extern "C" __global__ void {{kernel_name}}({%for arg in args%}{{arg.type}} {{'*' if arg.is_ptr}}{{arg.name}}, {% endfor %}
  {{index_type}} *row_offsets,
  {{index_type}} *eids,
  {{index_type}} *column_indices,
  {{index_type}} num_nodes,
  {{index_type}} max_dimx,
  {{index_type}} max_dimy,
  {{index_type}} tile_sizex,
  {{index_type}} tile_sizey) {
    for ({{index_type}} {{row_offset}} = blockIdx.y; {{row_offset}} < num_nodes; {{row_offset}} += gridDim.y) {
        {{index_type}} start_off = row_offsets[{{row_offset}}];
        {{index_type}} end_off = row_offsets[{{row_offset}} + 1];
        {{index_type}} blk_tid = threadIdx.x + threadIdx.y * blockDim.x;
        {{index_type}} tid = blk_tid + blockIdx.x * blockDim.x * blockDim.y;
        {{index_type}} warp_id = tid/32;
        {{index_type}} warp_tid = tid%32;
        {{index_type}} num_warp_x = max_dimx/tile_sizex;
        {{index_type}} tx = warp_tid % tile_sizex + warp_id % num_warp_x * tile_sizex; 
        {{index_type}} ty = warp_tid / tile_sizex + warp_id / num_warp_x * tile_sizey;
        for (;tx < max_dimx; tx += blockDim.x*gridDim.x) {
            for (;ty < max_dimy; ty += blockDim.y) {
                {%for agg_stmt in aggs%}{{agg_stmt.init}}{%endfor%}
                {{init_outter_offset}}
                for ({{index_type}} e=start_off;e<end_off;++e) {
                    {{index_type}} {{col_index}} = column_indices[e];
                    {{index_type}} eid = eids[e];
                    {{init_inner_offset}}
                    {%for edge_stmt in edges%}
                    {{edge_stmt.load}}
                    {{edge_stmt.compute}}
                    {{edge_stmt.inner_write}}{%endfor%}
                    {%for agg_stmt in aggs%}
                    {{agg_stmt.compute}}
                    {{agg_stmt.inner_write}}{%endfor%}
                }
                {%for agg_stmt in aggs%}
                {{agg_stmt.outter_write}}{%endfor%}
                {%for node_stmt in nodes%}
                {{node_stmt.load}}{{node_stmt.compute}}{{node_stmt.inner_write}}{%endfor%}
            }
        }
    }
}
"""
)

tpl_fa = Template(
"""
extern "C" __global__ void {{kernel_name}}({%for arg in args%}{{arg.type}} {{'*' if arg.is_ptr}}{{arg.name}}, {% endfor %}
  {{index_type}} *row_offsets,
  {{index_type}} *eids,
  {{index_type}} *column_indices,
  {{index_type}} num_nodes,
  {{index_type}} max_dimx,
  {{index_type}} max_dimy,
  {{index_type}} thrs_per_group,
  {{index_type}} nodes_per_block) {
    {{index_type}} {{row_offset}} = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;;
    if ({{row_offset}} < num_nodes) {
        {{index_type}} feat_len = max_dimx * max_dimy;
        {{index_type}} beg = __ldg(row_offsets + {{row_offset}});
        {{index_type}} end = __ldg(row_offsets + {{row_offset}} + 1);
        {{index_type}} tx = threadIdx.x % thrs_per_group;
        for (; tx<feat_len; tx+=blockDim.x) {
            {%for agg_stmt in aggs%}{{agg_stmt.init}}{%endfor%}
            {{init_outter_offset}}
            for ({{index_type}} e=beg;e<end;++e) {
                {{index_type}} {{col_index}} = __ldg(column_indices + e);
                {{index_type}} eid = __ldg(eids + e);
                {{init_inner_offset}}
                {%for edge_stmt in edges%}
                {{edge_stmt.load}}
                {{edge_stmt.compute}}
                {{edge_stmt.inner_write}}{%endfor%}
                {%for agg_stmt in aggs%}
                {{agg_stmt.compute}}
                {{agg_stmt.inner_write}}{%endfor%}
            }
            {%for agg_stmt in aggs%}
            {{agg_stmt.outter_write}}{%endfor%}
            {%for node_stmt in nodes%}
            {{node_stmt.load}}{{node_stmt.compute}}{{node_stmt.inner_write}}{%endfor%}
        }
    }
}
"""
)

EdgeInfo = namedtuple('EdgeInfo', ['load', 'compute', 'inner_write'])
NodeInfo = namedtuple('NodeInfo', ['load', 'compute', 'inner_write'])
ArgInfo = namedtuple('ArgInfo', ['name', 'type', 'is_ptr'])
AggInfo = namedtuple('AggInfo', ['init', 'compute', 'inner_write', 'outter_write'])

def gen_cuda(configs):
    h = ''
    for config in configs:
        if config['template_name'] == 'fa':
            rendered_tpl = tpl_fa.render(**config)
        elif config['template_name'] == 'v2':
            rendered_tpl = tpl_v2.render(**config)
        else:
            raise NotImplementedError('Have not implement template for', configs['template_name'])
        h += rendered_tpl
    return compile_cuda(h)
