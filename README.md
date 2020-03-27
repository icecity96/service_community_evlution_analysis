# ICSOC2020 Service Community Evolution Tracking and Pattern Analysising in Service Ecosystem

## Data Description
In `data` dictionary, we release our data used in this paper. The release dataset collection 
and processing process can refer to our previous work.
```latex

```
Related code be found in:

* [MSEM-EventExtraction](https://github.com/icecity96/eventextraction)
* [MSEM-EvolutionaryRelationGenerate](https://github.com/icecity96/MSEM-EvolutionaryRelationGenerate)

`data/nodes.json`: Each line represents a node in the json format. Each node contains following fields:

| field | description | example | note |
| --- | --- | --- | --- |
| id | node unique id | ofo | - |
| label | node label | ofo | for service and stakeholder label is same as id|
| type | node type | Stakeholder | In this paper, we only use Stakeholder and Service |

`data/edges.json`: Each line represents an edge in the json format. Each edge contains following fields:

| field | description | example | note |
| --- | --- | --- | --- |
| id | edge unique id | .... | - |
| source | source node id | 9 |- |
| target | target node id | HappyCycle | - |
| timestamp | when the edge was created | 2019-08-12 | Structure relation will be set to 1999-01-01 |
| type | the edge type | evolutionary | HasX, structural or evolutionary |
| r | the edge semantics type | BelongTo | this attribute will be used to aging function |
| generated_from | where the edge generated from | 9 |this attribute only for evolutionary type edge |

Service ecosystem snapshots generation, static community detection, nodes' social position calculation
are time consuming. So we provide a copy of our intermediate results, which have been stored in `data/*.pkl`. 
You can use this data to save your time.