import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import { Link } from "react-router-dom";
import ReactEcharts from "echarts-for-react";


export default class TweetSpread extends Component {
    constructor(props) {
        super(props);
        this.state = {
            claim: props.claim,
            option: {
                xAxis: {
                    type: 'category',
                    name: 'Date',
                },
                yAxis: {
                    type: 'value',
                    name: 'Tree size',
                },
                series: [
                    {
                        data: [
                        ],
                        type: 'scatter',
                    }
                ],
            },
        };
        //this.getDocumentDetails();
        this.renderTweetSpreadPage = this.renderTweetSpreadPage.bind(this);
        this.requestData = this.requestData.bind(this);
    }

    componentDidMount() {
        this.requestData();
    }

    requestData() {
        const requestOptions = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                claim: this.props.claim,
            }),
        };
        fetch("/api/get-tweetspread", requestOptions)
            .then((response) => response.json())
            .then((data) => {
                console.log(data);
                //this.props.history.push("/document")
                let stance_dic = ["Refute", "Neutral", "Support"];
                let stance_color_dic  = ["rgb(255,69,0,0.8)", "rgb(119,136,153, 0.8)", "rgb(76, 187, 23, 0.8)"];
                this.setState({
                    option: {
                        tooltip: {
                            show: true,
                            trigger: 'item',
                            backgroundColor: 'rgba(255,255,255,0.7)',
                            extraCssText: 'width:350px; white-space:pre-wrap; text-align:"left"',
                            formatter: function (params) {
                                //console.log("tooltip checking", params);
                                var str = "";
                                str = `<div style="width:300px">`;
                                if (params.data) {
                                    str += `  <div><span style="font-weight: bold"> Time</span>: ${params.data[0]}</div>`;
                                    str += `  <div><span style="font-weight: bold"> Stance</span>: <span style="color:${stance_color_dic[params.data[8]]}">${stance_dic[params.data[8]]}</span></div>`;
                                    str += `  <div><span style="font-weight: bold"> Direct Spread</span>: ${params.data[2]}</div>`;
                                    str += `  <div><span style="font-weight: bold"> Total Spread</span>: ${params.data[1]} (Refute ${params.data[4]} - Neutral ${params.data[5]} - Support ${params.data[6]})</div>`;
                                    str += `  <div><span style="font-weight: bold"> Text</span>: ${params.data[7]}</div>`;
                                }
                                str += `</div>`;
                                return str
                            },
                        },
                        visualMap: [
                            {
                                left: 'right',
                                top: '10%',
                                dimension: 2,
                                min: 0,
                                max: 1000,
                                itemWidth: 30,
                                itemHeight: 120,
                                calculable: true,
                                precision: 0.1,
                                text: ['Direct Spread'],
                                textGap: 30,
                                inRange: {
                                    symbolSize: [10, 70]
                                },
                                outOfRange: {
                                    symbolSize: [10, 70],
                                    color: ['rgba(255,255,255,0.4)']
                                },
                                controller: {
                                    inRange: {
                                        color: ['#c23531']
                                    },
                                    outOfRange: {
                                        color: ['#999']
                                    }
                                }
                            },
                            {
                                left: 'right',
                                bottom: '5%',
                                dimension: 3,
                                min: -1,
                                max: 1,
                                itemHeight: 120,
                                text: ['Support', 'Refute'],
                                textGap: 30,
                                inRange: {
                                    color: ['red', '#f2c31a', '#24b7f2']
                                },
                            }
                        ],
                        series: [
                            {
                                name: this.state.claim,
                                type: 'scatter',
                                data: this.sortData(data),
                                itemStyle: {
                                    opacity: 0.8,
                                    shadowBlur: 10,
                                    shadowOffsetX: 0,
                                    shadowOffsetY: 0,
                                    shadowColor: 'rgba(0,0,0,0.3)'
                                },
                            }
                        ]
                    },
                });
            });
    }

    sortData(data) {
        console.log("data", data);
        let data_sorted = data.sort(function (a, b) {
            return new Date(a.tweet.time) - new Date(b.tweet.time)
        })
        let data_list = [];
        for (var i = 0; i < data_sorted.length; i++) {
            let data_i = data_sorted[i];
            let stance_prop = (data_i["support"] - data_i["refute"]) / data_i["total_spread"];
            let tweet_content = data_i["tweet"]["text"];
            let tweet_info = [data_i["tweet"]["time"], data_i["total_spread"], data_i["direct_spread"], stance_prop, data_i["support"], data_i["neutral"], data_i["refute"], tweet_content, data_i["tweet"]["stance"]];
            data_list.push(tweet_info);
        }
        console.log("data list", data_list);
        return data_list
    }

    getDataList(neg, neu, pos) {
        let data = [];
        if (neg > 0) {
            data.push({ value: neg, name: 'Refute', itemStyle: { color: 'rgba(184, 29, 10, .9)' } })
        }
        if (neu > 0) {
            data.push({ value: neu, name: 'Neutral', itemStyle: { color: 'rgba(239, 183, 0, .9)' } })
        }
        if (pos > 0) {
            data.push({ value: pos, name: 'Support', itemStyle: { color: 'rgba(0, 132, 80, .9)' } })
        }
        return data
    }

    stanceColor(stance, similarity) {
        var max_index = stance.indexOf(Math.max(...stance));
        var color = "#fff";
        if (max_index == 0) {
            color = "rgba(184, 29, 10,"
        } else if (max_index == 1) {
            color = "rgba(239, 183, 0,"
        } else {
            color = "rgba(0, 132, 80,"
        }
        color = color + String(similarity) + ")";
        return color
    }

    renderTweetSpreadPage() {
        return (
            <Grid container align="center" style={{ marginLeft: '5%', marginRight: '5%' }}>
                <ReactEcharts option={this.state.option} style={{ height: 600, width: '90vw' }} />
            </Grid>
        );
    }


    render() {
        return (
            this.renderTweetSpreadPage()
        );
    }
}
