import React, { Component } from "react";

export default class SearchQueue extends Component {
  constructor(props) {
    super(props);
    console.log("search query", props)
    this.handleSearchButtonPressed = this.handleSearchButtonPressed.bind(this);
    this.handleSearchButtonPressed();
  }

  handleSearchButtonPressed() {
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
        this.props.history.push("/documents")
      });
  }


  render() {
    return <p> Loading search results ... </p>;
  }
}
