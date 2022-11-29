import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import { Switch, Route, Link, Redirect } from "react-router-dom";
import ReactEcharts from "echarts-for-react";


export default class TweetCount extends Component {
    constructor(props) {
        super(props);
        this.state = {
            claim: "Chloroquine can cure coronavirus",
            option: {},
        };
        //this.getDocumentDetails();
        this.renderTweetCountPage = this.renderTweetCountPage.bind(this);
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
        fetch("/api/get-tweetcount", requestOptions)
            .then((response) => response.json())
            .then((data) => {
                //console.log(data);
                let sort_data = this.sortData(data);
                //this.props.history.push("/document")
                //let stance_dic = ["Refute", "Neutral", "Support"];
                this.setState({
                    option: {
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
                        xAxis: {
                            type: 'time',
                            name: 'Date',
                        },
                        yAxis: {
                            type: 'value',
                            name: '#Tweet',
                        },
                        tooltip: {
                            show: true,
                            trigger: 'item',
                            backgroundColor: 'rgba(255,255,255,0.7)',
                            extraCssText: 'width:350px; white-space:pre-wrap; text-align:"left"',
                            formatter: function (params) {
                                var str = "";
                                str = `<div style="width:200px, align:"left">`;
                                str += `  <div><span style="font-weight: bold"> Count</span>: ${params.data[1]}</div>`;
                                str += `  <div><span style="font-weight: bold"> Time</span>: ${params.data[0]}</div>`;
                                str += `</div>`;
                                return str
                            },
                        },
                        series: [
                            {
                                name: 'countTwitter',
                                stack: 'Total',
                                type: 'line',
                                data: sort_data,
                            },
                        ]
                    },
                })
            });
    }


    sortData(data) {
        //count, time
        console.log("TweetCount input data", data);

        let data_list = data.sort(function (a, b) {
            return new Date(a.time) - new Date(b.time)
        })
        //let dates = [];
        let output = [];
        let counts = 0;
        for (var i = 0; i < data.length; i++) {
            //dates.push(data_list[i].time);
            counts = counts + data_list[i].count;
            output.push([data_list[i].time, counts]);
        }
        //let output = { "date": dates, "count": counts };
        console.log("TweetCount output data", output);
        return output
    }

    renderTweetCountPage() {
        return (
            <Grid container align="center" style={{ marginLeft: '5%', marginRight: '5%' }}>
                <ReactEcharts option={this.state.option} style={{ height: '600px', width: '90%' }} />
            </Grid>
        );
    }


    render() {
        return (
            this.renderTweetCountPage()
        );
    }
}
