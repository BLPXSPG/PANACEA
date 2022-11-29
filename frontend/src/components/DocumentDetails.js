import React, { Component } from "react";
import { Grid, Button, Typography, Link } from "@material-ui/core";
import { EmailShareButton, EmailIcon, TwitterShareButton, TwitterIcon, FacebookShareButton, FacebookIcon} from 'react-share';
import ReactEcharts from "echarts-for-react";


export default class DocumentDetails extends Component {
  constructor(props) {
    super(props);
    this.state = {
      datatype: "",
      id: "",
      neg: "",
      neu: "",
      pos: "",
      sentence_list: "",
      sent_stance_list: "",
      sent_similarity_list: "",
      query: "",
      source: "",
      title: "",
      url: "https://www.bbc.com",
      sentence: "",
      option: {
        title: {
          text: 'Document relation to searched sentence?',
          left: 'center'
        },
        tooltip: {
          trigger: 'item'
        },
        legend: {
          orient: 'vertical',
          left: 'left'
        },
        series: [
          {
            name: 'Access From',
            type: 'pie',
            radius: '50%',
            data: [
              { value: 1048, name: 'Search Engine', itemStyle:{color: 'red'} },
              { value: 735, name: 'Direct' },
              { value: 580, name: 'Email' },
              { value: 484, name: 'Union Ads' },
              { value: 300, name: 'Video Ads' }
            ],
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ]
      },
    };
    this.id = this.props.match.params.id;
    //this.getDocumentDetails();
    this.renderDocumentDetailsPage = this.renderDocumentDetailsPage.bind(this);
  }

  componentDidMount() {
    fetch("/api/get-document-details" + "?id=" + this.id)
      .then((response) => response.json())
      .then((data) => {
        data = data[0]
        console.log(data)
        //this.props.history.push("/document")
        this.setState({
          datatype: data.datatype,
          id: data.id,
          neg: data.neg,
          neu: data.neu,
          pos: data.pos,
          sentence_list: JSON.parse(data.sentence_list),
          sent_stance_list: JSON.parse(data.sent_stance_list),
          sent_similarity_list: JSON.parse(data.sent_similarity_list),
          query: data.query,
          source: data.source,
          title: data.title,
          url: data.url,
          sentence: data.sentence,
          option: {
            title: {
              text: 'Document relation to searched sentence?',
              left: 'center'
            },
            tooltip: {
              trigger: 'item',
            },
            legend: {
              orient: 'vertical',
              top: '25',
              left: 'left'
            },
            series: [
              {
                name: 'Stance',
                type: 'pie',
                radius: ['40%', '70%'],
                top: '25',
                data:this.getDataList(data.neg, data.neu, data.pos),
                emphasis: {
                  itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                  }
                }
              }
            ]
          },
        });
      });
  }

  getDataList(neg, neu, pos) {
    let data = [];
    if (neg > 0) {
      data.push({value: neg, name: 'Refute', itemStyle: { color: 'rgba(184, 29, 10, .9)' } })
    }
    if (neu > 0) {
      data.push({ value: neu, name: 'Neutral', itemStyle: { color: 'rgba(119,136,153, .9)' } })
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
      color = "rgba(119,136,153,"
    } else {
      color = "rgba(0, 132, 80,"
    }
    color = color + String(similarity) + ")";
    return color
  }

  renderDocumentDetailsPage(){
    var sents = [<Typography variant="body1"/>];
    //console.log("after",this.state.defaultQuery);
    //console.log(this.state.doc_data)
    console.log(this.state.sentence_list.length, this.state.sent_stance_list.length, this.state.sent_similarity_list.length)
    for (var i = 0; i < this.state.sentence_list.length; i++) {
      sents.push(<span style={{ backgroundColor:this.stanceColor(this.state.sent_stance_list[i], this.state.sent_similarity_list[i]) }}>{this.state.sentence_list[i]} &nbsp;</span>);
    }

    return (
      <Grid container spacing={1}  className="homepage">
        <Grid item xs={12} align="center" style={{backgroundColor:'rgba(128, 128, 128, 0.1)', minHeight:100, marginBottom: 10}}>
          <Typography variant="h5" style={{marginTop:30}}> {this.state.title}</Typography>

          Full document in <a href={this.state.url} target="_blank">{this.state.source} </a> | Share to 

          <EmailShareButton
            title={this.state.title}
            url={window.location.href}
            className="shareBtn col-md-1 col-sm-1 col-xs-1">
            <EmailIcon size={24} round />
          </EmailShareButton>

          <TwitterShareButton
            title={this.state.title}
            url={window.location.href}
            className="shareBtn col-md-1 col-sm-1 col-xs-1">
            <TwitterIcon size={24} round />
          </TwitterShareButton>

          <FacebookShareButton
            title={this.state.title}
            url={window.location.href}
            className="shareBtn col-md-1 col-sm-1 col-xs-1">
            <FacebookIcon size={24} round />
          </FacebookShareButton>
        </Grid>
        <Grid container align="center" style={{marginLeft:'5%', marginRight:'5%'}}>
          <Grid item xs={8} align="left">
            <Typography style={{fontSize:20}}> Article related to: {this.state.query}</Typography>
            <p style={{fontSize:20, marginTop:10, marginBottom:5}}>Document Content:</p>
            <p style={{color:'rgba(68, 68, 68, 1)'}}>{sents}</p>
          </Grid>
          <Grid item xs={4} align="center" style={{minHeight:350}}>
            <ReactEcharts option={this.state.option} />
          </Grid>

        </Grid>

        <div>
          <div className="footerStyle">© 2022 Copyright: Warwick NLP Group and QUML Cognitive Science Research Group</div>
        </div>
      </Grid>

    );
  }

  
  render() {
    return (
      this.renderDocumentDetailsPage()
    );
  }
}
