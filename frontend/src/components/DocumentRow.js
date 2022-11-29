import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import { Link } from "react-router-dom";
import ReactEcharts from "echarts-for-react";

export default class SentenceRow extends Component {
  constructor(props) {
    super(props);
    console.log("inside", props);

    this.state = {
      //id = props.obj.id,
      similarity: props.similarity,
      stance: props.obj.stance, 
      title: props.obj.title,
      similarity: Math.exp(props.obj.docscore).toFixed(2),
      abstract: props.obj.content.substr(0, 500),
      datatype: props.obj.datatype,
      neg: props.obj.neg,
      neu: props.obj.neu,
      pos: props.obj.pos,
      source: props.obj.source,
      url: props.obj.url,
      id: props.obj.id,
      path_to_doc: "/document/"+String(props.obj.id),
      color: "purple",
      option: {
        xAxis: {
          type: 'category',
          data: ['Ref', 'Neu', 'Sup']
        },
        yAxis: {
          show: false,
        },
        series: [
          {
            data: [
              { value: props.obj.neg, name: 'Ref', itemStyle: { color: 'rgba(184, 29, 10, .9)' } },
              { value: props.obj.neu, name: 'Neu', itemStyle: { color: 'rgba(119,136,153, .9)' } },
              { value: props.obj.pos, name: 'Sup', itemStyle: { color: 'rgba(0, 132, 80, .9)' } },
            ],
            type: 'bar',
          }
        ],
      },
    };

    this.toDocumentDetailsPage = this.toDocumentDetailsPage.bind(this);
    this.renderSentenceRow = this.renderSentenceRow.bind(this);
  }

  getDataList(neg, neu, pos) {
    let data = [];
    if (neg > 0) {
      data.push({value: neg, name: 'Ref', itemStyle: { color: 'rgba(184, 29, 10, .9)' } })
    }
    if (neu > 0) {
      data.push({ value: neu, name: 'Neu', itemStyle: { color: 'rgba(119,136,153, .9)' } }) //yellow 239, 183, 0
    }
    if (pos > 0) {
      data.push({ value: pos, name: 'Sup', itemStyle: { color: 'rgba(0, 132, 80, .9)' } })
    }
    return data
  }

  stanceColor(stance) {
    var color = "#fff";
    if (stance == "Refute") {
      color = "rgba(184, 29, 10, .9)"
    } else if (stance == "Neutral") {
      color = "rgba(119,136,153, .9)"
    } else {
      color = "rgba(0, 132, 80, .9)"
    }
    return color
  }

  toDocumentDetailsPage() {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: this.state.id,
      }),
    };
    fetch("/api/get-document-details", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log(data)
        //this.props.history.push("/document")
      });
  }


  renderSentenceRow() {
    return (
      <Grid container spacing={1} className="document-row" style={{ padding: 5, backgroundColor: "rgb(193, 197, 205, 0.3)", border: "none"}} >
        <Grid item xs={9} align="center">
          <Grid item xs={12} align="left" style={{ marginTop: 10, marginLeft: 20, padding: 5,}}>
            <Button
              color="grey"
              variant="contained"
              style={{ marginTop: 10, backgroundColor:'rgba(255, 255, 255, 0)', border: 'none', boxShadow:'none', fontWeight:'bold', padding: 0,}}
              //onClick={this.toDocumentDetailsPage}
              to={this.state.path_to_doc}
              component={Link}
            >
              {this.state.title}
            </Button>
          </Grid>

          <Grid item xs={12} align="left" style={{color:'rgba(128, 128, 128, 1)', marginLeft: 20}}>
            <Typography>... {this.state.abstract} ...</Typography>
          </Grid>

          <Grid item xs={12} align="left" style={{color:'rgba(128, 128, 128, 1)', fontSize:12, marginTop:20, marginLeft: 20}}>
            <a href={this.state.url} target="_blank">https://{this.state.url.split('/')[2]}/{this.state.url.split('/')[3]} </a>
          </Grid>
          
        </Grid>

        <Grid item xs={3} align="center">
          <Grid item xs={12} align="right" style={{ marginTop: 15 }}>
            <Typography className="document-label" style={{fontSize:12}}>TYPE: {this.state.datatype}</Typography>
          </Grid>
          <Grid item xs={12} align="right" style={{ marginTop: 10 }}>
            <Typography className="document-label" style={{fontSize:12}}>SOURCE: {this.state.source}</Typography>
          </Grid>
          <Grid item xs={12} align="right" style={{ marginTop: 10, color:this.state.color }}>
            <Typography className="document-label" style={{fontSize:12}}>RELEVANCE SCORE: {this.state.similarity}</Typography>
          </Grid>
          <Grid item xs={12} align="right" style={{ marginTop: 10,}}>
            <Typography className="document-label" style={{height: 18, fontSize:12}}>STANCE:&nbsp;<p style={{ color:this.stanceColor(this.state.stance) }}>{this.state.stance}</p></Typography>
          </Grid>
          <Grid item xs={12} align="right" style={{ marginBottom: 15, marginLeft:20}}>
            <ReactEcharts option={this.state.option} style={{height: '100px', maxWidth:'200px'}} />
          </Grid>
        </Grid>
      </Grid>
    );
  }
  render() {
    return (
      this.renderSentenceRow()
    );
  }
}
