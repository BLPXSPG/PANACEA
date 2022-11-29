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


export default class TweetVocab extends Component {
    constructor(props) {
        super(props);
        this.state = {
            claim: props.claim,
            stance:props.stance,
            //date: "2020-03-19",
            option: {
                xAxis: {
                    type: 'category',
                    name: 'Date',
                    show: false
                },
                yAxis: {
                    type: 'value',
                    name: 'Total Spread',
                    show: false
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
        this.renderTweetVocabPage = this.renderTweetVocabPage.bind(this);
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
                stance: this.props.stance,
            }),
        };
        fetch("/api/get-vocabulary", requestOptions)
            .then((response) => response.json())
            .then((data) => {
                //console.log(data);
                let sort_data = this.sortData(data);
                //this.props.history.push("/document")
                //let stance_dic = ["Refute", "Neutral", "Support"];
                let claim = this.props.stance;
                this.setState({
                    option: {
                        tooltip: {
                            trigger: 'item'
                        },
                        legend: {
                            orient: 'vertical',
                            left: 'left'
                        },
                        tooltip: {
                            show: true,
                            trigger: 'item',
                            backgroundColor: 'rgba(255,255,255,0.7)',
                            extraCssText: 'width:350px; white-space:pre-wrap; text-align:"left"',
                            formatter: function (params) {
                                var str = "";
                                str = `<div style="width:200px, align:"left">`;
                                if (params.data.time) {
                                    str += `  <div>Time: ${params.data.time}</div>`;
                                }
                                str += `</div>`;
                                return str
                            },
                        },
                        series: [
                            {
                                type: 'wordCloud',
                                tooltip: { show: false },
                                sizeRange: [30, 80],
                                rotationRange: [0, 0],
                                gridSize: 8,
                                shape: 'square',
                                data: sort_data,
                                textStyle: {
                                    fontFamily: 'sans-serif',
                                    fontWeight: 'bold',
                                    // Color can be a callback function or a color string
                                    color: function () {
                                        // Random color
                                        if (claim==0) {
                                            return 'rgb(' + [
                                                Math.round(Math.random() * 255),
                                                Math.round(Math.random() * 30),
                                                Math.round(Math.random() * 30)
                                            ].join(',') + ')';
                                        } else {
                                            return 'rgb(' + [
                                                Math.round(Math.random() * 30),
                                                Math.round(Math.random() * 255),
                                                Math.round(Math.random() * 30)
                                            ].join(',') + ')';
                                        }
                                        
                                    }
                                },
                            },
                        ]
                    },
                })
            });
    }


    sortData(data) {
        //word, count
        console.log("TweetVocab input data", data);

        let topic_list = [];
        for (var i = 0; i < data.length; i++) {
            let word = { "name": data[i].word, "value": data[i].count };
            topic_list.push(word);
        }
        console.log("TweetVocab output data", topic_list);
        return topic_list
    }

    renderTweetVocabPage() {
        return (
            <Grid container align="center" style={{ marginLeft: '5%', marginRight: '5%' }}>
                <ReactEcharts option={this.state.option} style={{ height: '70vh', width: '90%' }} />
            </Grid>
        );
    }


    render() {
        return (
            this.renderTweetVocabPage()
        );
    }
}
