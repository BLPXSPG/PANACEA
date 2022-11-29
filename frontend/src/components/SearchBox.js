import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import Autocomplete from "./Autocomplete";
import TweetChoice from "./TweetChoice";


export default class SearchBox extends Component {
  defaultQuery = "vitamin C cures COVID-19";
  constructor(props) {
    super(props);

    this.state = {
      queryInput: this.defaultQuery,
      loadstate: " ",
      querychoices: ['drinking lemon water prevents COVID-19',
        'CPR Still Encouraged During COVID-19 Pandemic',
        'masks impact the coronavirus transmission',
        'vitamin C cures COVID-19',],
      claim_id_dic: {},
    };
    this.handleQueryChange = this.handleQueryChange.bind(this);
    this.getClaim = this.getClaim.bind(this);
  }

  componentDidMount() {
    this.getClaim();
  }

  getClaim() {
    fetch("/api/get-claim")
      .then((response) => response.json())
      .then((data) => {
        let claims = [];
        let claim_id_dic = {};
        for (var i = 0; i < data.length; i++) {
          claims.push(data[i].claim);
          claim_id_dic[data[i].claim] = data[i].id;
        }
        this.setState({
          querychoices: claims,
          claim_id_dic: claim_id_dic,
        });
        console.log("claim get", this.state.querychoices);
      });
  }

  handleQueryChange(e) {
    const REACT_VERSION = React.version;
    console.log(e, REACT_VERSION);
    this.setState({
      //queryInput: e.target.textContent,
      queryInput: e,
    });
    console.log(this.state.queryInput)
  }

  renderSearchBox() {
    //const classes = useStyles();
    return (
      <Grid container spacing={1} alignItems="center" justify="center">

        <Grid container className="background-fog" style={{ height: "60vh", backgroundPosition: "top" }} >
          <Grid item xs={1} align="center" style={{ padding: 20, marginTop: "10%" }}>
            <div className="logo-warwick" href="www.bbc.com"></div>
            <div className="logo-qm"></div>
            <div className="logo-ukri"></div>
          </Grid>
          <Grid item xs={10} align="center" style={{ padding: 20, marginLeft: "3vw", color: "white", marginTop: "25vh" }}>
            <Grid item align="center" style={{ width: "70%" }}>
              <Typography variant="h1" compact="h1">
                <a href="https://panacea2020.github.io/" style={{ color: "white", textDecoration: 'none', fontWeight: '800' }}>PANACEA</a>
              </Typography>
              <Typography variant="h5" compact="h5">
                <div align="center">
                  <span style={{ color: "rgba(84, 108, 160)", fontWeight: '800' }}>PAN</span>demic{" "}
                  <span style={{ color: "rgba(84, 108, 160)", fontWeight: '800' }}>A</span>i{" "}
                  <span style={{ color: "rgba(84, 108, 160)", fontWeight: '800' }}>C</span>laim{" "}
                  v<span style={{ color: "rgba(84, 108, 160)", fontWeight: '800' }}>E</span>racity{" "}
                  <span style={{ color: "rgba(84, 108, 160)", fontWeight: '800' }}>A</span>ssessment</div>
              </Typography>
            </Grid>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid container style={{ padding: 20, backgroundColor: "rgb(193, 197, 205, 0.3)", marginTop: 5 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid item xs={10} align="center" style={{ padding: 20 }}>
            <Grid item xs={10} align="center">
              <b style={{ padding: 30, fontSize:30, fontWeight: '800' }}> FACT CHECKING </b>
            </Grid>
            <Grid container>
              <Grid item xs={4} style={{ padding: 20 }}>
                <div style={{ fontWeight: 'bold', fontSize: 18, color: "rgba(84, 108, 160)" }}>
                  <i class="fa fa-check-square-o" style={{ width: 25 }}></i>
                  Fact Checking
                </div>
                <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                  PANACEA allows users to input a natural language claim and do veracity assessment by incorporating evidence from trustable external sources, such as scientific publications.
                </div>
              </Grid>
              <Grid item xs={4} style={{ padding: 20 }}>
                <div style={{ fontWeight: 'bold', fontSize: 18, color: "rgba(84, 108, 160)" }}>
                  <i class="fa fa-folder-open-o" style={{ width: 25 }}></i>
                  Evidences
                </div>
                <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                  The veracity assessment (True/False) would be run by our pre-trained model online. Supporting evidence would be selected and ranked. The results can be filtered by the source, stance and relevance of each supporting evidence towards the claim.
                </div>
              </Grid>
              <Grid item xs={4} style={{ padding: 20 }}>
                <div style={{ fontWeight: 'bold', fontSize: 18, color: "rgba(84, 108, 160)" }}>
                  <i class="fa fa-pie-chart" style={{ width: 25 }}></i>
                  Analysis
                </div>
                <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                  PANACEA also provides detailed analysis, such as the stance distribution of supporting evidence towards the input claim. Charts and highlighted key sentences are used for better user visualisation.                </div>
              </Grid>
            </Grid>
            <Grid item xs={10} align="center">
              <Autocomplete
                suggestions={this.state.querychoices}
                path={this.props.path}
              />
            </Grid>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "20vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent:"center", alignItems:"center" }}>
              <span style={{ display: "table-cell", verticalAlign: "middle" }}>
                During the COVID-19 pandemic, national and international organisations are using social media and online platforms to communicate information about the virus to the public. However, propagation of misinformation has also become prevalent. This can strongly influence human behaviour and negatively impact public health interventions, so it is vital to detect misinformation in a timely manner. For instance, unreliable treatments could put public safety in danger and increase pressure on the health system; failure to comply with government advice may increase the chance of spreading the disease. Veracity assessment of online information is a complex problem, we need to adopt a sophisticated conceptual framework combining techniques and algorithms from natural language processing, knowledge graphs, network analysis, deep learning and visual informatics.
              </span>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid container style={{ padding: 20, backgroundColor: "rgb(193, 197, 205, 0.3)", marginTop: 5 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid item xs={10} align="center" style={{ padding: 20 }}>
            <Grid item xs={10} align="center">
              <b style={{ padding: 30, fontSize:30, fontWeight: '800' }}> RUMOUR DETECTION </b>
            </Grid>
            <Grid container>
              <Grid item xs={4} style={{ padding: 20 }}>
                <div style={{ fontWeight: 'bold', fontSize: 18, color: "rgba(84, 108, 160)" }}>
                  <i class="fa fa-search" style={{ width: 25 }}></i>
                  Rumour Detection
                </div>
                <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                  PANACEA allows users to input a natural language claim and do rumour detection (True/False/Unverified) by its related tweets’ propagation pattern.
                </div>
              </Grid>
              <Grid item xs={4} style={{ padding: 20 }}>
                <div style={{ fontWeight: 'bold', fontSize: 18, color: "rgba(84, 108, 160)" }}>
                  <i class="fa fa-sitemap" style={{ width: 25 }}></i>
                  Database
                </div>
                <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                  Besides a small annotated dataset with common rumours, an active crawler also collects Covid-19 related streaming at the backend to maintain an updated database.
                </div>
              </Grid>
              <Grid item xs={4} style={{ padding: 20 }}>
                <div style={{ fontWeight: 'bold', fontSize: 18, color: "rgba(84, 108, 160)" }}>
                  <i class="fa fa-pie-chart" style={{ width: 25 }}></i>
                  Analysis
                </div>
                <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                  Detailed analysis including stance towards the source tweet are generated. Various propagation statistics, including the word cloud, propagation tree, spread map, etc., are visualised for better user experience.
                </div>
              </Grid>
            </Grid>
            <Grid item xs={10} align="center">
              <TweetChoice
                suggestions={this.state.querychoices}
                path={this.props.path}
                claim_id_dic={this.state.claim_id_dic}
              />
            </Grid>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>



      </Grid>
    );
  }

  render() {
    return (
      this.renderSearchBox()
    );
  }
}

