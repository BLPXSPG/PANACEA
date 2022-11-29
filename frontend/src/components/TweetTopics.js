import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import { Switch, Route, Link, Redirect } from "react-router-dom";
import SearchBox from "./SearchBox";
import TweetSpread from "./TweetSpread";
import TweetGraph from "./TweetGraph";
import ReactEcharts from "echarts-for-react";
import world from 'echarts/asset/world.json';
import * as echarts from 'echarts';
import 'echarts-wordcloud';


export default class TweetTopic extends Component {
    constructor(props) {
        super(props);
        this.state = {
            claim: props.claim,
            option: {
                xAxis: {
                    type: 'value',
                    name: 'x',
                },
                yAxis: {
                    type: 'value',
                    name: 'y',
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
        this.renderTweetTopicPage = this.renderTweetTopicPage.bind(this);
        this.requestData = this.requestData.bind(this);
        this.onUserClick = this.onUserClick.bind(this);

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
        fetch("/api/get-tweettopic", requestOptions)
            .then((response) => response.json())
            .then((data) => {
                //console.log(data);
                console.log("TweetTopic input data", data);
                //this.props.history.push("/document")
                //let stance_dic = ["Refute", "Neutral", "Support"];
                let sort_data = this.sortData(data);
                this.setState({
                    option: {
                        tooltip: {
                            show: true,
                            trigger: 'item',
                            backgroundColor: 'rgba(255,255,255,0.7)',
                            extraCssText: 'width:350px; white-space:pre-wrap; text-align:"left"',
                            formatter: function (params) {
                                var str = "";
                                str = `<div style="width:200px, align:"left">`;
                                if (params.data[3]) {
                                    str += `  <div><span style="font-weight: bold"> Coordinates</span>: (${params.data[0]}, ${params.data[1]})</div>`;
                                    str += `  <div><span style="font-weight: bold"> Weight</span>: ${params.data[2]}</div>`;
                                    str += `  <div><span style="font-weight: bold"> Representative</span>: ${params.data[3]}</div>`;
                                } else {
                                    str += `  <div><span style="font-weight: bold"> Weight</span>: ${params.data}</div>`;
                                }
                                str += `</div>`;
                                return str
                            },
                        },
                        xAxis: [
                            { type: 'value', gridIndex: 0 },
                            { type: 'value', gridIndex: 1 }
                        ],
                        yAxis: [
                            { type: 'value', gridIndex: 0 },
                            { type: 'category', gridIndex: 1 }
                        ],
                        grid: [{ bottom: '55%' }, { top: '55%' }],
                        series: [
                            {
                                name: "Topic",
                                type: 'scatter',
                                data: sort_data,
                                symbolSize: function (params) {
                                    return params[2] * 500;
                                }
                            },
                            {
                                name: "Top Words",
                                type: 'bar',
                                data: [],
                                itemStyle: { color: '#8399A8' },
                                showBackground: true,
                                backgroundStyle: {
                                    color: 'rgba(180, 180, 180, 0.2)'
                                }
                            },

                        ]
                    },
                });
            });
    }

    sortData(data) {
        let data_list = [];
        for (var i = 0; i < data.length; i++) {
            let data_i = data[i];
            let tweet_info = [data_i["x"], data_i["y"], data_i["weight"], data_i["text"], data_i["topicwords"]];
            data_list.push(tweet_info);
        }
        console.log("data list", data_list);
        return data_list
    }

    onUserClick = (params) => {
        console.log("Echarts click", params)
        let wordlist = [];
        let wordweight = [];
        for (var i = 0; i < params.data[4].length; i++) {
            wordlist.push([params.data[4][i].word]);
            wordweight.push([params.data[4][i].weight] * 100);
        }
        this.setState({
            option: {
                xAxis: [
                    { type: 'value', gridIndex: 0 },
                    { type: 'value', gridIndex: 1 }
                ],
                yAxis: [
                    { type: 'value', gridIndex: 0 },
                    { type: 'category', gridIndex: 1, data: wordlist.reverse()}
                ],
                grid: [{ bottom: '55%' }, { top: '55%' }],
                series: [
                    {
                        name: "Top Words",
                        type: 'bar',
                        data: wordweight.reverse(),
                        itemStyle: { color: '#8399A8' },
                        showBackground: true,
                        backgroundStyle: {
                            color: 'rgba(180, 180, 180, 0.2)'
                        },
                        xAxisIndex: 1,
                        yAxisIndex: 1,
                    },

                ]
            },
        });
    }


    renderTweetTopicPage() {
        return (
            <Grid container align="center" style={{ marginLeft: '5%', marginRight: '5%' }}>
                <ReactEcharts
                    option={this.state.option}
                    style={{ height: '70vh', width: '90%' }}
                    ref={(e) => { this.echarts_react = e; }}
                    onEvents={{
                        'click': this.onUserClick,
                    }} />
            </Grid>
        );
    }


    render() {
        return (
            this.renderTweetTopicPage()
        );
    }
}
