var React = require('react')
var ReactDOM = require('react-dom')

class App extends React.Component {
  render() {
    return (
      <div>
        <h1>react loes me</h1>
      </div>
    );
  }
}

ReactDOM.render(<App/>, document.getElementById('app'))
