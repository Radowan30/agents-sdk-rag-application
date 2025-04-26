import PromptForm from './components/PromptForm';

function App() {
  return (
    <div className="bg-gray-50 min-h-screen">
      <header className="fixed top-0 left-0 right-0 bg-white shadow-md z-50 p-4">
        <h1 className="text-2xl font-bold text-center">📝 RAG App AI 🤖</h1>
        <p className="text-center text-sm text-gray-600 mt-1">Talk with your documents</p>
      </header>
      <PromptForm />
    </div>
  );
}

export default App;
