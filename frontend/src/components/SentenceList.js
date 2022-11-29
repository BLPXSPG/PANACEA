import React, { Component } from "react";
import { Link } from "react-router-dom";
import SentenceRow from "./SentenceRow";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import Navbar from "./Navbar";
export default class SentenceList extends Component {

  constructor(props) {
    super(props);
    this.state = {
      queryInput: " ",
      n_docs: 0,
      doc_data: [],
      original_data: [],
      veracity: " ",
      source_filterlist: [],
      activeFilter: [],
      veracity_confidence: " ",
      filterscore: 0,
      filterstance: 0,
      loadstate: " ",
    };

    this.handleSearchButtonPressed = this.handleSearchButtonPressed.bind(this);
    //this.getDocumentDetails = this.getDocumentDetails.bind(this);
    this.handleQueryChange = this.handleQueryChange.bind(this);
    //this.getDocumentDetails();
    this.onFilterChange = this.onFilterChange.bind(this);
    this.renderSentenceListPage = this.renderSentenceListPage.bind(this);
    this.reRank = this.reRank.bind(this);
  }

  componentDidMount() {
    fetch("/api/get-sentence")
      .then((response) => response.json())
      .then((data) => {
        data.sort(function (a, b) {
          return b.similarity - a.similarity
        });
        console.log("load data", data);
        this.setState({
          queryInput: data[0].documents[0].query,
          veracity: data[0].documents[0].veracity,
          veracity_confidence: data[0].documents[0].veracity_confidence,
          n_docs: data.length,
          doc_data: data,
          original_data: data,
          source_filterlist: this.getSourceList(data, data.length),
          activeFilter: [],
        })
      });
  }

  loadingPage() {
    return (
      <Grid item xs={12} align="center">
        <div class="loader"></div>
        Running model for the results ...
      </Grid>
    )
  }

  handleSearchButtonPressed() {
    this.setState({ loadstate: this.loadingPage() });

    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: this.state.queryInput,
      }),
    };
    fetch("/api/input-query", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log("data get", data)
        console.log("now to the updated page");
        //this.props.history.push("/sentences")
        window.location.reload();
      });
  }

  getSourceList(data, n_docs) {
    let source_filterlist = [];
    for (var i = 0; i < n_docs; i++) {
      //console.log(data[i]["documents"])
      source_filterlist.push(data[i]["documents"][0]["source"]);
    }
    let source_set = [...new Set(source_filterlist)];
    var filterList = [];
    for (var j = 0; j < source_set.length; j++) {
      filterList.push({ "id": j, "source": source_set[j] });
    }
    return filterList
  }

  reRank(rerankby) {
    let rerank_data = [...this.state.original_data];
    if (rerankby === 1) {
      rerank_data.sort((a, b) => (a.neg > b.neg) ? 1 : ((b.neg > a.neg) ? -1 : 0));
      rerank_data.reverse();
    } else if (rerankby === 2) {
      rerank_data.sort((a, b) => (a.neu > b.neu) ? 1 : ((b.neu > a.neu) ? -1 : 0));
      rerank_data.reverse();
    } else if (rerankby === 3) {
      rerank_data.sort((a, b) => (a.pos > b.pos) ? 1 : ((b.pos > a.pos) ? -1 : 0));
      rerank_data.reverse();
    } else {
    }
    this.setState({
      doc_data: rerank_data
    })
  }

  handleQueryChange(e) {
    this.setState({
      queryInput: e.target.value,
    });
  }

  stanceColor(stance, similarity) {
    var color = "#fff";
    if (stance == false) {
      color = "rgba(184, 29, 10,";
    } else if (stance == true) {
      color = "rgba(0, 132, 80,";
      similarity = 1 - similarity;
    }
    color = color + String(similarity) + ")";
    return color
  }

  onFilterChange(filter) {
    const filterList = this.state.source_filterlist;
    const activeFilter = this.state.activeFilter;
    //const { filterList, activeFilter } = this.state;
    if (filter === "ALL") {
      if (activeFilter.length === filterList.length) {
        this.setState({ activeFilter: [] });
      } else {
        this.setState({ activeFilter: this.state.source_filterlist.map(filter => filter.source) });
      }
    } else {
      if (activeFilter.includes(filter)) {
        const filterIndex = activeFilter.indexOf(filter);
        const newFilter = [...activeFilter];
        newFilter.splice(filterIndex, 1);
        this.setState({ activeFilter: newFilter });
      } else {
        this.setState({ activeFilter: [...activeFilter, filter] });
      }
    }
  }


  onRelevanceChange(score) {
    this.setState({ filterscore: score })
  }

  onStanceChange(stance) {
    this.setState({ filterstance: stance })
  }

  renderSentenceListPage() {
    const filterList = this.state.source_filterlist;
    const activeFilter = this.state.activeFilter;
    let filteredList;
    if (
      activeFilter.length === 0 ||
      activeFilter.length === filterList.length
    ) {
      filteredList = this.state.doc_data;
    } else {
      filteredList = this.state.doc_data.filter(item =>
        this.state.activeFilter.includes(item.documents[0].source)
      );
    }
    //console.log("filteredList", filteredList, activeFilter)

    let filteredSource;
    if (this.state.filterscore === 0.6) {
      filteredSource = filteredList.filter(item =>
        item.similarity >= 0.6)
    } else if (this.state.filterscore === 0.3) {
      filteredSource = filteredList.filter(item =>
        item.similarity >= 0.3)
    } else {
      filteredSource = filteredList;
    }
    //console.log("filteredSource", filteredSource, typeof filteredSource)

    let filteredStance;
    if (this.state.filterstance === 1) {
      filteredStance = filteredSource.filter(item =>
        item.stance === "Refute")
    } else if (this.state.filterstance === 2) {
      filteredStance = filteredSource.filter(item =>
        item.stance === "Neutral")
    } else if (this.state.filterstance === 3) {
      filteredStance = filteredSource.filter(item =>
        item.stance === "Support")
    } else {
      filteredStance = filteredSource
    }
    console.log("filteredStance", filteredStance, typeof filteredStance)

    let docs = [];
    if (Array.isArray(filteredStance)) {
      docs = filteredStance.map(item =>
        <SentenceRow obj={item.documents[0]} similarity={item.similarity} stance={item.stance} sentence={item.sentence} key={item.id} />
      );
    }

    //for (var filtered_doc in filteredList) {
    //console.log("filtered list i", filtered_doc);
    //docs.push(<SentenceRow obj={filteredList[filtered_doc]} key={filtered_doc} />);
    //console.log("docs", docs)
    //}
    return (
      <Grid container className="homepage">

        <Navbar></Navbar>

        <Grid item xs={12} align="center" className="background-fog-flex" style={{ height: "20vh", backgroundPosition: "top", minHeight: 200 }}>
          <Typography variant="h4" compact="h4" style={{ marginTop: '3vh', padding: 15, color: "white" }}>
            <div align="center">
              <span style={{ fontWeight: '800' }}> Input Claim: </span>
              <span> {this.state.queryInput} </span>
            </div>
            <Grid align="center" style={{ color: this.stanceColor(this.state.veracity, this.state.veracity_confidence), fontWeight: '700' }}>
              {this.state.veracity.toString().toUpperCase()} <span style={{ fontSize: 20 }}></span>
            </Grid>
          </Typography>
          <Grid item xs={12} align="center" style={{ fontWeight: "bold", color: "rgba(255, 255, 255, 0.9)" }}>
            <p>{this.state.loadstate}</p>
          </Grid>
          <div align="center">
            <span style={{ color: "rgba(255, 255, 255, 0.8)" }}> Showing top <span style={{ fontWeight: '600', color: "white" }}> 30 </span> supporting sentences found </span>
          </div>
        </Grid>

        <Grid container style={{ marginLeft: 20, marginRight: 20 }}>
          <Grid item xs={2} align="left">
            <Typography style={{ fontSize: 20, marginTop: 30 }}> &#8694; Show Type:</Typography>
            <Grid style={{ marginLeft: 20 }}>
              <Button
                color="grey"
                variant="contained"
                style={{ marginTop: 10 }}
                to="/documents"
                component={Link}
              >
                Article
              </Button>
              <Button
                color="grey"
                variant="contained"
                style={{ marginTop: 10, backgroundColor: "rgba(128, 128, 128, 0.8)", color: "white" }}
                to="/sentences"
                component={Link}
              >
                Sentence
              </Button>
            </Grid>
            <Typography style={{ fontSize: 20, marginTop: 20 }}> &#8694; Sort by:</Typography>
            <div>
              <form style={{ marginLeft: 20, fontSize: 15, color: 'grey' }}>
                <label htmlFor="polaritynone">Relevance</label>
                <input
                  id="polaritynone"
                  type="radio"
                  name="polarity"
                  onClick={() => this.reRank(0)}
                />
                <br />
                <label htmlFor="polarityneg">Refute Stance</label>
                <input
                  id="polarityneg"
                  type="radio"
                  name="polarity"
                  onClick={() => this.reRank(1)}
                />
                <br />
                <label htmlFor="polarityneu">Neutral Stance</label>
                <input
                  id="polarityneu"
                  type="radio"
                  name="polarity"
                  onClick={() => this.reRank(2)}
                />
                <br />
                <label htmlFor="polaritypos">Support Stance</label>
                <input
                  id="polaritypos"
                  type="radio"
                  name="polarity"
                  onClick={() => this.reRank(3)}
                />
              </form>
            </div>

            <Typography style={{ fontSize: 20, marginTop: 20 }}> &#8694; Filter:</Typography>
            <Typography style={{ fontSize: 15, marginLeft: 20 }}>Source Filter</Typography>
            <div>
              <form style={{ marginLeft: 40, fontSize: 15, color: 'grey' }}>
                <label htmlFor="sourceInput">All</label>
                <input
                  id="sourceInput"
                  type="checkbox"
                  onClick={() => this.onFilterChange("ALL")}
                  checked={this.state.activeFilter.length === this.state.source_filterlist.length}
                />
                {this.state.source_filterlist.map(filter => (

                  <React.Fragment>
                    <div>
                      <label htmlFor={filter.id}>{filter.source}</label>
                      <input
                        id={filter.id}
                        type="checkbox"
                        checked={this.state.activeFilter.includes(filter.source)}
                        onClick={() => this.onFilterChange(filter.source)}
                      />
                    </div>
                  </React.Fragment>
                ))}
              </form>
            </div>

            <Typography style={{ fontSize: 15, marginLeft: 20 }}>Relevance Filter</Typography>
            <div>
              <form style={{ marginLeft: 40, fontSize: 15, color: 'grey' }}>
                <label htmlFor="relevanceall">All</label>
                <input
                  id="relevanceall"
                  type="radio"
                  name="relevance"
                  onClick={() => this.onRelevanceChange(0)}
                />
                <br />
                <label htmlFor="relevancet">Score above 30%</label>
                <input
                  id="relevancet"
                  type="radio"
                  name="relevance"
                  onClick={() => this.onRelevanceChange(0.3)}
                />
                <br />
                <label htmlFor="relevances">Score above 60%</label>
                <input
                  id="relevances"
                  type="radio"
                  name="relevance"
                  onClick={() => this.onRelevanceChange(0.6)}
                />
              </form>
            </div>

            <Typography style={{ fontSize: 15, marginLeft: 20 }}>Stance Filter</Typography>
            <div>
              <form style={{ marginLeft: 40, fontSize: 15, color: 'grey' }}>
                <label htmlFor="stanceall">All</label>
                <input
                  id="stanceall"
                  type="radio"
                  name="stance"
                  onClick={() => this.onStanceChange(0)}
                />
                <br />
                <label htmlFor="stanceneg">Refute</label>
                <input
                  id="stanceneg"
                  type="radio"
                  name="stance"
                  onClick={() => this.onStanceChange(1)}
                />
                <br />
                <label htmlFor="stanceneu">Neutral</label>
                <input
                  id="stanceneu"
                  type="radio"
                  name="stance"
                  onClick={() => this.onStanceChange(2)}
                />
                <br />
                <label htmlFor="stancepos">Support</label>
                <input
                  id="stancepos"
                  type="radio"
                  name="stance"
                  onClick={() => this.onStanceChange(3)}
                />
              </form>
            </div>

          </Grid>
          <Grid item xs={10} align="center" style={{ marginTop: 20, }}>
            <Grid item xs={10} align="center" style={{ marginRight: 30, }}>

              {docs}
            </Grid>
          </Grid>
        </Grid>

        <div>
          <div className="footerStyle">© 2022 Copyright: Warwick NLP Group and QUML Cognitive Science Research Group</div>
        </div>

      </Grid>


    )

  }

  render() {
    return (
      this.renderSentenceListPage()
    );
  }
}
