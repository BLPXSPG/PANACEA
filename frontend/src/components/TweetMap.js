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


export default class TweetMap extends Component {
  constructor(props) {
    super(props);
    console.log("check props claim", props.claim,)
    this.state = {
      claim: props.claim,
      option: {
        xAxis: {
          type: 'category',
          name: 'Date',
          show:false
        },
        yAxis: {
          type: 'value',
          name: 'Total Spread',
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
    this.renderTweetMapPage = this.renderTweetMapPage.bind(this);
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
    fetch("/api/get-tweetmap", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        let sort_data = this.sortData(data);
        //this.props.history.push("/document")
        //
        this.setState({
          option: this.Map(sort_data),
        })
      });
  }

  Map(data) {
    echarts.registerMap("world", world);
    //console.log("get current data", current_data);
    let stance_color_dic = ["rgb(255,69,0,0.8)", "rgb(119,136,153, 0.8)", "rgb(76, 187, 23, 0.8)"]; //["rgb(255,69,0,0.8)", "rgb(255, 165, 0, 0.8)", "rgb(30,144,255, 0.8)"];
    let stance_dic = ["Refute", "Neutral", "Support"];

    const opt = {
      geo: {
        map: 'world',
        //top: "30%",
        z: 0,
        roam: true,
        tooltip: { show: true },
        scaleLimit: {
          min: 0.5,
          max: 5,
        },
      },
      xAxis: {
        show:false
      },
      yAxis: {
        show:false
      },
      //grid: { bottom: '75%', right: '35%', },
      tooltip: {
        show: true,
        trigger: 'item',
        formatter: function (params) {
          var str = "";
          str = `<div style="min-width:100px">`;
          if (params.data.time) {
            str += `  <div><span style="font-weight: bold"> Time</span>: ${params.data.time}</div>`;
            str += `  <div><span style="font-weight: bold"> Text</span>: ${params.data.text}</div>`;
            str += `  <div><span style="font-weight: bold"> Location</span>: ${params.data.location}</div>`;
            str += `  <div><span style="font-weight: bold"> Stance</span>: <span style="color:${stance_color_dic[params.data.stance]}">${stance_dic[params.data.stance]}</span></div>`;
          }
          str += `</div>`;
          return str
        },
        extraCssText: 'box-shadow: 0 0 3px transparent;',
        //borderColor: 'transparent',
        //textStyle: { color: 'white' },
        //backgroundColor: 'rgba(163, 163, 194, 0.9)',
        enterable: true,
        alwaysShowContent: true,
      },
      legend:{
        show: true,
        data: ['Refute', 'Neutral', 'Support'],
      },
      series: [
        {
          name: 'Refute',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          map: 'world',
          scaleLimit: {
            min: 0.5,
            max: 5,
          },
          itemStyle: {
            color: "rgb(255,69,0,0.8)",
            //color: function(params){
            //  return stance_color_dic[params.data.stance]
            //},
          },
          label: {
            normal: {
              show: false,
              position: 'top',
              formatter: '  {b}',
              color: 'grey',
              fontWeight: "bold",
              fontSize: 15,
            }
          },
          symbolSize: 6,
          data: data[0],
          rippleEffect: {
            period: 5,
            scale: 5,
            brushType: 'fill'
          },
        },
        {
          name: 'Neutral',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          map: 'world',
          scaleLimit: {
            min: 0.5,
            max: 5,
          },
          itemStyle: {
            color: "rgb(119,136,153, 0.8)",
            //color: function(params){
            //  return stance_color_dic[params.data.stance]
            //},
          },
          label: {
            normal: {
              show: false,
              position: 'top',
              formatter: '  {b}',
              color: 'grey',
              fontWeight: "bold",
              fontSize: 15,
            }
          },
          symbolSize: 6,
          data: data[1],
          rippleEffect: {
            period: 5,
            scale: 5,
            brushType: 'fill'
          },
        },
        {
          name: 'Support',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          map: 'world',
          scaleLimit: {
            min: 0.5,
            max: 5,
          },
          itemStyle: {
            color: "rgb(76, 187, 23, 0.8)",
            //color: function(params){
            //  return stance_color_dic[params.data.stance]
            //},
          },
          label: {
            normal: {
              show: false,
              position: 'top',
              formatter: '  {b}',
              color: 'grey',
              fontWeight: "bold",
              fontSize: 15,
            }
          },
          symbolSize: 6,
          data: data[2],
          rippleEffect: {
            period: 5,
            scale: 5,
            brushType: 'fill'
          },
        },
      ],
    };
    return opt
  }

  sortData(data) {
    console.log("tweetmap input data", data);
    let data_list_pos = [];
    let data_list_neu = [];
    let data_list_neg = [];
    for (var i = 0; i < data.length; i++) {
      let data_i = data[i];
      let tweet_info = {"name": data_i.id, "location":data_i.tweet.location, "value":[data_i.tweet.longitude, data_i.tweet.latitude], "time":data_i.time, "stance":data_i.tweet.stance, "text":data_i.tweet.text};
      if (data_i.tweet.stance==2){
        data_list_pos.push(tweet_info);
      } else if (data_i.tweet.stance==1){
        data_list_neu.push(tweet_info);
      } else {
        data_list_neg.push(tweet_info);
      }
    }
    //console.log("tweetmap output data", data_list);
    return [data_list_neg, data_list_neu, data_list_pos]
  }

  getDataList(neg, neu, pos) {
    let data = [];
    if (neg > 0) {
      data.push({ value: neg, name: 'Refute', itemStyle: { color: 'rgba(184, 29, 10, .9)' } })
    }
    if (neu > 0) {
      data.push({ value: neu, name: 'Neutral', itemStyle: { color: 'rgba(119,136,153, .9)' } }) //yellow 239, 183, 0 //grey 119,136,153 //blue 0, 132, 80
    }
    if (pos > 0) {
      data.push({ value: pos, name: 'Support', itemStyle: { color: 'rgba(76, 187, 23, .9)' } })//Green - 76, 187, 23
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

  renderTweetMapPage() {
    return (
        <Grid container align="center" style={{ marginLeft: '5%', marginRight: '5%' }}>
            <ReactEcharts option={this.state.option} style={{ height: '70vh', width: '90vw' }} />
        </Grid>
    );
  }


  render() {
    return (
      this.renderTweetMapPage()
    );
  }
}
