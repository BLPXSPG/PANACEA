import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import { Link } from "react-router-dom";
import ReactEcharts from "echarts-for-react";


export default class TweetGraph extends Component {
    constructor(props) {
        super(props);
        this.state = {
            claim: "Chloroquine can cure coronavirus",
            sort_data: { "nodes": [], "links": [], "categories": [] },
            option: {
                tooltip: {
                    trigger: 'item'
                },
                dataZoom: [
                    {
                        type: 'inside',
                    },
                    {
                    }
                ],
                legend: {
                    orient: 'vertical',
                    left: 'left'
                },
                series: [
                    {
                        name: 'Propagation graph',
                        type: 'graph',
                        layout: 'circular',
                        data: [{ "id": 1, "category": 0 }, { "id": 2, "category": 0 }, { "id": 3, "category": 1 }],
                        links: [{ "source": 1, "target": 2 }, { "source": 3, "target": 2 }],
                        categories: [{ "name": "A" }, { "name": "B" }],
                        roam: true,
                        label: {
                            show: false,
                            position: 'right',
                            formatter: '{b}'
                        },
                        labelLayout: {
                            hideOverlap: true
                        },
                        scaleLimit: {
                            min: 0.4,
                            max: 2
                        },
                        lineStyle: {
                            color: 'source',
                            curveness: 0.3
                        }
                    }
                ]
            },
        };
        //this.getDocumentDetails();
        this.renderTweetGraphPage = this.renderTweetGraphPage.bind(this);
        this.requestData = this.requestData.bind(this);
        this.sortData = this.sortData.bind(this);
        this.updateOption = this.updateOption.bind(this);

        //let sort_data = this.requestData();
        //this.updateOption(sort_data);

    }

    componentDidMount() {
        //Promise.all([this.requestData(),])
        //.then((value) => {
        //    this.updateOption(value);
        //})
        this.requestData();
        //this.updateOption(value);
    }

    requestData() {
        const requestOptions = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                claim: this.props.claim,
            }),
        };
        Promise.all([
            fetch("/api/get-claim").then(response => response.json()),
            fetch("/api/get-tweetgraph", requestOptions).then(response => response.json()),
        ])
            .then((value) => {
                console.log("value get", value);
                let sort_data = this.sortData(value);
                console.log("value get and cleaned", sort_data);
                this.updateOption(sort_data);
            })
    }


    updateOption(sort_data) {
        console.log("sort data", sort_data);
        let stance_dic = ["Refute", "Neutral", "Support"];
        let stance_color_dic = ["rgb(255,69,0,0.8)", "rgb(119,136,153, 0.8)", "rgb(76, 187, 23, 0.8)"];
        this.setState({
            sort_data: sort_data,
            option: {
                xAxis: {
                    type: 'time',
                    //data: sort_data.dates,
                },
                yAxis: {
                    type: 'value',
                    scale: true,
                    show: false,
                },
                legend: {
                    show:false
                },
                tooltip: {
                    //arr = {"id":node_index, "x": node_l["time"], "y": l/date_k.length, "comment_count": node_l["comment_count"], "retweet_count": node_l["retweet_count"], "spread_type": node_l["spread_type"], "category": node_l["stance"], "text":node_l["text"]};
                    show: true,
                    trigger: 'item',
                    backgroundColor: 'rgba(255,255,255,0.7)',
                    extraCssText: 'width:350px; white-space:pre-wrap; text-align:"left"',
                    formatter: function (params) {
                        var str = "";
                        str = `<div style="width:200px, align:"left">`;
                        if (params.data[4]) {
                            str += `  <div><span style="font-weight: bold"> Time</span>: ${params.data[0]}</div>`;
                            str += `  <div><span style="font-weight: bold"> Stance</span>: <span style="color:${stance_color_dic[params.data[5]]}">${stance_dic[params.data[5]]}</span></div>`;
                            str += `  <div><span style="font-weight: bold"> Type</span>: ${params.data[4]}</div>`;
                            str += `  <div><span style="font-weight: bold"> ID</span>: ${params.data[7]}</div>`;
                            str += `  <div><span style="font-weight: bold"> Text</span>: ${params.data[6]}</div>`;
                            str += `  <div style="margin-top:10px">`;
                            str += `    <span style="padding-left"> <i class="fa fa-retweet"></i> ${params.data[3]}</span>`;
                            str += `    <span style="padding-right"> <i class="fa fa-comment"></i> ${params.data[2]}</span>`;
                            str += `  </div>`;
                        }
                        if (params.data.source) {
                            str += `  <div style="margin-top:10px">`;
                            str += `    <span style="padding-left"> From ${params.data.source}</span>`;
                            str += `    <span style="padding-right"> To ${params.data.target}</span>`;
                            str += `  </div>`;
                        }
                        str += `</div>`;
                        return str
                    },
                },
                series: [{
                    //name: claims[i].claim,
                    type: 'graph',
                    coordinateSystem: 'cartesian2d',
                    edgeSymbol: ['none', 'arrow'],
                    data: sort_data.data,
                    links: sort_data.links,
                    categories: sort_data.categories,
                    roam: true,
                    emphasis:{focus:'adjacency'},
                    label: {
                        show: false,
                        position: 'right',
                        formatter: '{b}'
                    },
                    labelLayout: {
                        hideOverlap: true
                    },
                    scaleLimit: {
                        min: 0.4,
                        max: 2
                    },
                    itemStyle: {
                        color: function(params){
                          return stance_color_dic[params.data[5]]
                        },
                    },
                    lineStyle: {
                        color: 'source',
                        curveness: 0.3
                    }
                }],
            },
        });
    }

    sortData(data) {
        let tweetinfo = data[1]
        let categories = [{"name":"Refute"}, {"name":"Neutral"}, {"name":"Support"}, ];
        //for (var i = 0; i < claims.length; i++) {
            //console.log(i, claims[i])
            // for given claim
        let nodes = [];
        let links = [];
        let check_exist = [];
        let node_index = 0;
        let id_dic = {};
        for (var j = 0; j < tweetinfo.length; j++) {
            let data_j = tweetinfo[j];
            let source = data_j["source"];
            let target = data_j["target"];
            if (!(check_exist.includes(source["id"]))) {
                check_exist.push(source["id"])
                nodes.push([source["time"], Math.random(), source["comment_count"], source["retweet_count"], source["spread_type"], source["stance"], source["text"], node_index]);
                id_dic[source["id"]] = node_index;
                node_index = node_index + 1;
            }
            // save target nodes
            if (!(check_exist.includes(target["id"]))) {
                check_exist.push(target["id"])
                nodes.push([target["time"], Math.random(), target["comment_count"], target["retweet_count"], target["spread_type"], target["stance"], target["text"], node_index]);
                id_dic[target["id"]] = node_index;
                node_index = node_index + 1;
            }
            //links.push({ "source": source["id"], "target": target["id"] });
            links.push({ "source": id_dic[source["id"]], "target": id_dic[target["id"]] });
        }
        let total_data = { "data": nodes, "links": links, "categories": categories}
        return total_data
    }

    sortData_old(data) {
        //let claims = data[0]
        let tweetinfo = data[1]
        //console.log("data", data);
        //let category_i = 0;
        let categories = [{"name":"Refute"}, {"name":"Neutral"}, {"name":"Support"}, ];
        let dates = [];
        let series = [];
        //for (var i = 0; i < claims.length; i++) {
            //console.log(i, claims[i])
            // for given claim
        let nodes = [];
        let links = [];
        let check_exist = [];
        let stance_color_dic = ["rgb(255,69,0,0.8)", "rgb(119,136,153, 0.8)", "rgb(76, 187, 23, 0.8)"];
        for (var j = 0; j < tweetinfo.length; j++) {
            let data_j = tweetinfo[j];
            let source = data_j["source"];
            let target = data_j["target"];
            if (!(check_exist.includes(source["id"]))) {
                check_exist.push(source["id"])
                nodes.push(source);
                dates.push(source["time"]);
            }
            // save target nodes
            if (!(check_exist.includes(source["id"]))) {
                check_exist.push(target["id"])
                nodes.push(target);
                dates.push(target["time"]);
            }
            links.push({ "source": source["id"], "target": target["id"] });
        }
        let node_index = 0;
        let id_dic = {};
        dates = [...new Set(dates)];
        let date_sorted = dates.sort(function (a, b) {
            return new Date(a) - new Date(b)
        });
        let node_withy = [];
        let link_reorder = [];
        if (nodes.length != 0) {
            for (var k = 0; k < date_sorted.length; k++) {
                const date_k = nodes.filter(node => node["time"]==date_sorted[k]);
                //console.log(date_k, dates[k])
                for (var l = 0; l < date_k.length; l++){
                    let node_l = date_k[l];
                    //let arr = {"id":node_index, "x": node_l["time"], "y": l/date_k.length, "comment_count": node_l["comment_count"], "retweet_count": node_l["retweet_count"], "spread_type": node_l["spread_type"], "category": node_l["stance"], "text":node_l["text"]};
                    let arr = [node_l["time"], Math.random(), node_l["comment_count"], node_l["retweet_count"], node_l["spread_type"], node_l["stance"], node_l["text"], node_index];
                    id_dic[node_l["id"]] = node_index;
                    node_index = node_index + 1;
                    //arr.splice(1,0,(l/date_k.length));
                    node_withy.push(arr);
                }
            }
            for (var m = 0; m < links.length; m++){
                link_reorder.push({ "source": id_dic[links[m]["source"]], "target": id_dic[links[m]["target"]] });
            }
            //categories.push(claims[i].claim);
            series.push({
                //name: claims[i].claim,
                type: 'graph',
                coordinateSystem: 'cartesian2d',
                edgeSymbol: ['none', 'arrow'],
                data: node_withy,
                links: link_reorder,
                categories: categories,
                roam: true,
                emphasis:{focus:'adjacency'},
                label: {
                    show: false,
                    position: 'right',
                    formatter: '{b}'
                },
                labelLayout: {
                    hideOverlap: true
                },
                scaleLimit: {
                    min: 0.4,
                    max: 2
                },
                itemStyle: {
                    color: function(params){
                      return stance_color_dic[params.data[5]]
                    },
                },
                lineStyle: {
                    color: 'source',
                    curveness: 0.3
                }
            });
        }
        let total_data = { "data": node_withy, "links": link_reorder, "dates": date_sorted, "categories": categories}
        return total_data
    }


    renderTweetGraphPage() {
        return (
            <Grid container align="center" style={{ marginLeft: '5%', marginRight: '5%' }}>
                <ReactEcharts option={this.state.option} style={{ height: '70vh', width: '90vw' }} />
            </Grid>

        );
    }


    render() {
        return (
            this.renderTweetGraphPage()
        );
    }
}
