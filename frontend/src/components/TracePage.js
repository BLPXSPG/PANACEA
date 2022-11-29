import React, { Component, useRef, useEffect } from "react";
import { Grid, Button, Typography } from "@material-ui/core";
import { Switch, Route, Link, Redirect } from "react-router-dom";
import TweetSpread from "./TweetSpread";
import TweetGraph from "./TweetGraph";
import TweetCount from "./TweetCount";
import 'echarts-wordcloud';
import TweetMap from "./TweetMap";
import TweetVocab from "./TweetVocab";
import Navbar from "./Navbar";
import TweetTopic from "./TweetTopics";

const useMountEffect = fun => useEffect(fun, []);

export default class TracePage extends Component {
  //defaultQuery = "Chloroquine can cure coronavirus";
  constructor(props) {
    super(props);
    this.countRef = React.createRef();
    this.wordRef = React.createRef();
    this.topicRef = React.createRef();
    this.spreadRef = React.createRef();
    this.graphRef = React.createRef();
    this.mapRef = React.createRef();
    console.log("props", props, props.match.params.id);
    this.state = {
      queryInput: this.props.match.params.id,
      loadstate: " ",
      loading_info: "test",
      n_tweets: 132,
      veracity_confidence: 0.9,
      veracity: "False",
      //chosen_date: '2020-03-19',
    };
    this.id = this.props.match.params.id;
    this.renderTracePage = this.renderTracePage.bind(this);
    this.claimSelection = this.claimSelection.bind(this);
    this.loadingPage = this.loadingPage.bind(this);
    this.getClaim = this.getClaim.bind(this);
  }

  componentDidMount() {
    this.getClaim();
  }

  getClaim() {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claim: this.props.match.params.id,
      }),
    };
    fetch("/api/get-claiminfo", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log("claim get at trace page", data)
        this.setState({
          queryInput: data["claim"],
          n_tweets: data["data_count"],
        });

        //for (var i = 0; i < data.length; i++) {
        //if (this.props.match.params.id == data[i].id) {
        //this.setState({
        //queryInput: data[i].claim,
        //});
        //}
        //}
      });
  }

  claimSelection() {
    console.log("update query and chosen date",)
    this.setState({
      loading_info: 'estimate time: 1 min'
    }, () => {
      this.setState({
        loadstate: this.loadingPage()
      });
    });
  }

  scrollTo(ref) {
    if (!ref.current) return;
    ref.current.scrollIntoView({ behavior: "smooth" });
  }

  loadingPage() {
    return (
      <Grid item xs={12} align="center">
        <div class="loader"></div>
        Running model for the results ...
        {this.state.loading_info}
      </Grid>
    )
  }

  stanceColor(stance, similarity) {
    var color = "#fff";
    if (stance == "Rumour") {
      color = "rgba(184, 29, 10,";
    } else if (stance == "Not Rumour") {
      color = "rgba(0, 132, 80,";
    } else if (stance == "Unverified") {
      color = "rgba(255, 204, 0,"
    }
    color = color + String(similarity) + ")";
    return color
  }



  renderTracePage() {
    return (
      <div className="homepage">
        <Navbar></Navbar>

        <Grid container className="background-fog" style={{ height: "70vh", backgroundPosition: "top" }} >
          <Grid item xs={1} align="center" style={{ padding: 20, marginTop: "5%" }}>
            <div className="logo-warwick" href="www.bbc.com"></div>
            <div className="logo-qm"></div>
            <div className="logo-ukri"></div>
          </Grid>
          <Grid item xs={10} align="center" style={{ padding: 20, marginLeft: "3vw", color: "white", marginTop: "10vh" }}>
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
              <Typography variant="h4" compact="h4" style={{ marginTop: '10vh', padding: 20 }}>
                <div align="center">
                  <span style={{ fontWeight: '800' }}> Input Claim: </span>
                  <span> {this.state.queryInput} </span>
                </div>
                <Grid align="center" style={{ color: this.stanceColor(this.state.veracity, this.state.veracity_confidence), fontWeight: '700' }}>
                  {this.state.veracity.toUpperCase()} <span style={{ fontSize: 20 }}></span>
                </Grid>
              </Typography>
              <div align="center">
                <span style={{ color: "rgba(255, 255, 255, 0.8)" }}> Showing <span style={{ fontWeight: '600', color: "white" }}> {this.state.n_tweets} </span>  related tweets found </span>
              </div>
            </Grid>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid container style={{ padding: 30, backgroundColor: "rgb(193, 197, 205, 0.3)", marginTop: 5, minHeight: "30vh", }}>
          <Grid item xs={2} style={{ padding: 20 }}>
            <button className="charttype" onClick={() => this.scrollTo(this.countRef)}>
              <div style={{ fontWeight: 'bold', fontSize: 50, color: "rgba(84, 108, 160)" }}>
                <i class="fa fa-twitter"></i>
              </div>
              <div style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(84, 108, 160)" }}>
                Tweet Count
              </div>
              <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                The total number of tweets related to the input claim against the posting date
              </div>
            </button>
          </Grid>
          <Grid item xs={2} style={{ padding: 20 }}>
            <button className="charttype" onClick={() => this.scrollTo(this.wordRef)}>
              <div style={{ fontWeight: 'bold', fontSize: 50, color: "rgba(84, 108, 160)" }}>
                <i class="fa fa-cloud"></i>
              </div>
              <div style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(84, 108, 160)" }}>
                Word Cloud
              </div>
              <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                Highlight popular words in tweets with negative or positive stance towards input claim
              </div>
            </button>
          </Grid>
          <Grid item xs={2} style={{ padding: 20 }}>
            <button className="charttype" onClick={() => this.scrollTo(this.topicRef)}>
              <div style={{ fontWeight: 'bold', fontSize: 50, color: "rgba(84, 108, 160)" }}>
                <i class="fa fa-pencil"></i>
              </div>
              <div style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(84, 108, 160)" }}>
                Discussion Topics
              </div>
              <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                Topics discussed among tweets related to the input claim
              </div>
            </button>
          </Grid>
          <Grid item xs={2} style={{ padding: 20 }}>
            <button className="charttype" onClick={() => this.scrollTo(this.spreadRef)}>
              <div style={{ fontWeight: 'bold', fontSize: 50, color: "rgba(84, 108, 160)" }}>
                <i class="fa fa-comments"></i>
              </div>
              <div style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(84, 108, 160)" }}>
                Source Tweet Influence
              </div>
              <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                The influence of each original tweet by the number of its comments/retweets
              </div>
            </button>
          </Grid>
          <Grid item xs={2} style={{ padding: 20 }}>
            <button className="charttype" onClick={() => this.scrollTo(this.graphRef)}>
              <div style={{ fontWeight: 'bold', fontSize: 50, color: "rgba(84, 108, 160)" }}>
                <i class="fa fa-sitemap"></i>
              </div>
              <div style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(84, 108, 160)" }}>
                Propagation Graph
              </div>
              <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                The propagation graph link the source tweet and its comments/retweets
              </div>
            </button>
          </Grid>
          <Grid item xs={2} style={{ padding: 20 }}>
            <button className="charttype" onClick={() => this.scrollTo(this.mapRef)}>
              <div style={{ fontWeight: 'bold', fontSize: 50, color: "rgba(84, 108, 160)" }}>
                <i class="fa fa-globe"></i>
              </div>
              <div style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(84, 108, 160)" }}>
                Tweet Map
              </div>
              <div style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 13 }}>
                Geo location of related tweets on map colored by their stance
              </div>
            </button>
          </Grid>
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "20vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div ref={this.countRef} style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ fontWeight: 'bold', fontSize: 25, color: "rgba(23, 35, 54)" }}>
                <i class="fa fa-twitter"></i>
                Tweet Count
                <span style={{ fontWeight: 'bold', fontSize: 15 }}> - show the total influence and discussion of the input claim. </span>
                <br />
                <span style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 15, color: "rgba(238, 241, 246)" }}>
                  The total number of tweets related to the input claim against the posting date are shown here.
                </span>
              </div>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid>
          <TweetCount claim={this.state.queryInput} />
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "20vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div ref={this.wordRef} style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ fontWeight: 'bold', fontSize: 25, color: "rgba(23, 35, 54)" }}>
                <i class="fa fa-cloud"></i>
                Word Cloud
                <span style={{ fontWeight: 'bold', fontSize: 15 }}> - show the popular topics of the input claim on both sides.</span>
                <br />
                <span style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 15, color: "rgba(238, 241, 246)" }}>
                  The top 30 words in tweets <span style={{ fontWeight: 'bold' }}>refute</span> the input claim and the top 30 words in tweets <span style={{ fontWeight: 'bold' }}>support</span> the input claim are shown in word cloud. Stopwords, punctuation, numbers are removed to reduce the noninformative words.
                </span>
              </div>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid style={{ height: "80vh" }}>
          <Grid container style={{ height: '60vh', width: '90vw' }}>
            <Grid item xs={1}>
            </Grid>
            <Grid item xs={5}>
              <span style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(23, 35, 54)" }}> Negative</span>
              <TweetVocab claim={this.state.queryInput} stance={0} />
            </Grid>
            <Grid item xs={5}>
              <span style={{ fontWeight: 'bold', fontSize: 20, color: "rgba(23, 35, 54)" }}> Positive</span>
              <TweetVocab claim={this.state.queryInput} stance={2} />
            </Grid>
            <Grid item xs={1}>
            </Grid>
          </Grid>
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "15vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div ref={this.topicRef} style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ fontWeight: 'bold', fontSize: 25, color: "rgba(23, 35, 54)" }}>
                <i class="fa fa-globe"></i>
                Discussion Topics
                <span style={{ fontWeight: 'bold', fontSize: 15 }}> - show the stance and popularity of the input claim in the different regions.</span>
                <br />
                <span style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 15, color: "rgba(238, 241, 246)" }}>
                  For each topics discussed related to the input claim, we have plot it below with coordinate being the PCA of the topic representation and raduis being the weight of the topic. 
                  When you select any topic, the detailed information will show in tooltip including representative sentence.
                  The top 10 words of selected topic will also presented with the corresponding weight.
                </span>
              </div>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid style={{ height: "80vh" }}>
          <TweetTopic claim={this.state.queryInput} />
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "20vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div ref={this.spreadRef} style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ fontWeight: 'bold', fontSize: 25, color: "rgba(23, 35, 54)" }}>
                <i class="fa fa-comments"></i>
                Source Tweet Influence
                <span style={{ fontWeight: 'bold', fontSize: 15 }}> - show the impact of each original tweet.</span>
                <br />
                <span style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 15, color: "rgba(238, 241, 246)" }}>
                  The influence of each original tweet are shown in the scatter, whereas the radius denotes the number of tweets that are directly commented to or retweeted from the original tweet. 
                  The y-axis “Tree size” also includes the number of indirect comments/retweets of the original tweet, such as the retweets of retweets, etc.
                </span>
              </div>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid style={{ height: "80vh" }}>
          <TweetSpread claim={this.state.queryInput} />
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "20vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div ref={this.graphRef} style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ fontWeight: 'bold', fontSize: 25, color: "rgba(23, 35, 54)" }}>
                <i class="fa fa-sitemap"></i>
                Propagation Graph
                <span style={{ fontWeight: 'bold', fontSize: 15 }}> - show the difference in propagation pattern between rumour and non-rumour tweets. </span>
                <br />
                <span style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 15, color: "rgba(238, 241, 246)" }}>
                  The propagation graph between the source tweet and its comments/retweets, showing the information spread path.
                </span>
              </div>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid style={{ height: "80vh" }}>
          <TweetGraph claim={this.state.queryInput} />
        </Grid>

        <Grid container align="center" className="background-fog" style={{ color: "white", marginTop: 5, height: "15vh", backgroundPosition: "bottom", minHeight: 150 }}>
          <Grid item xs={1} align="center"></Grid>
          <Grid xs={10} align="center">
            <div ref={this.mapRef} style={{ color: "white", fontSize: 16, fontWeight: 'normal', textAlign: 'left', padding: 10, marginBottom: '50', height: 150, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ fontWeight: 'bold', fontSize: 25, color: "rgba(23, 35, 54)" }}>
                <i class="fa fa-globe"></i>
                Tweet Map
                <span style={{ fontWeight: 'bold', fontSize: 15 }}> - show the stance and popularity of the input claim in the different regions.</span>
                <br />
                <span style={{ fontWeight: 'normal', textAlign: 'left', fontSize: 15, color: "rgba(238, 241, 246)" }}>
                  Related tweets of input claim are plotted on the world map and colored by their stances, where red/grey/green represents refute/neutral/support. The difference of stance and popularity towards the input claim in the different regions can be easily seen, which shows the local concern and geo location bias.
                </span>
              </div>
            </div>
          </Grid>
          <Grid item xs={1} align="center"></Grid>
        </Grid>

        <Grid style={{ height: "80vh" }}>
          <TweetMap claim={this.state.queryInput} />
        </Grid>




        <div>
          <div className="footerStyle">© 2022 Copyright: Warwick NLP Group and QUML Cognitive Science Research Group</div>
        </div>

      </div>
    );
  }

  render() {
    return (
      this.renderTracePage()
    );
  }

}

