package dev.danvega.sbdocs;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import jakarta.annotation.PostConstruct;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.ExtractedTextFormatter;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntSupplier;

@Component
public class ReferenceDocsLoader {

    private static final Logger log = LoggerFactory.getLogger(ReferenceDocsLoader.class);
    private final JdbcClient jdbcClient;
    private final VectorStore vectorStore;

    public ReferenceDocsLoader(JdbcClient jdbcClient, VectorStore vectorStore) {
        this.jdbcClient = jdbcClient;
        this.vectorStore = vectorStore;
    }

    @PostConstruct
    public void init() {

        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        // Assuming your PDF files are directly under classpath:/docs/
        Resource[] resources = new Resource[]{};
        try {
            resources = resolver.getResources("classpath:/docs/*.pdf");
        } catch (IOException e) {
            log.error("Unable to load pdf(s)", e);
        }
        List<Resource> pdfResources = Arrays.asList(resources);

        IntSupplier recordsCount = () -> jdbcClient.sql("select count(*) from vector_store")
                .query(Integer.class)
                .single();
        log.info("Current count of the Vector Store: {}", recordsCount.getAsInt());
        if (recordsCount.getAsInt() == 0) {

            log.info("Loading documents");
            //loadUsingStanfordNLP(pdfResources);
            loadUsingSpring(pdfResources);
            log.info("Current count of the Vector Store: {}", recordsCount.getAsInt());
        }
        log.info("Application is ready");
    }

    private void loadUsingSpring(List<Resource> pdfResources) {
        try (ForkJoinPool customPool = new ForkJoinPool(16)) {
            for (Resource pdfResource : pdfResources) {
                var config = PdfDocumentReaderConfig.builder()
                        .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder().withNumberOfBottomTextLinesToDelete(0)
                                .withNumberOfTopPagesToSkipBeforeDelete(0)
                                .build())
                        .withPagesPerDocument(1)
                        .build();

                try {
                    var pdfReader = new PagePdfDocumentReader(pdfResource, config);
                    var textSplitter = new TokenTextSplitter();
                    customPool.submit(() -> chunkList(textSplitter.apply(pdfReader.get()), 10).parallelStream()
                                    .forEach(vectorStore::accept))
                            .get();
                } catch (Exception e) {
                    log.warn("{} failed to load", pdfResource.getFilename());
                }
            }
        }
    }

    private void loadUsingStanfordNLP(List<Resource> pdfResources) {
        ForkJoinPool customPool = new ForkJoinPool(16);
        try {
            Properties props = new Properties();
            props.setProperty("annotators", "tokenize, ssplit");
            props.setProperty("threads", "256"); // Set the desired number of threads
            props.setProperty("coref.algorithm", "neural");

            StanfordCoreNLP pipeline = new StanfordCoreNLP(props, false);
            pdfResources.forEach(pdfResource -> {
                try {
                    log.info("Processing {}", pdfResource.getFilename());
                    PDDocument pdDocument = Loader.loadPDF(pdfResource.getContentAsByteArray());
                    PDFTextStripper stripper = new PDFTextStripper();
                    String text = stripper.getText(pdDocument);
                    pdDocument.close();
                    log.info("{} {} text length", pdfResource.getFilename(), text.length());

                    edu.stanford.nlp.pipeline.CoreDocument coreDocument = new edu.stanford.nlp.pipeline.CoreDocument(text);
                    pipeline.annotate(coreDocument);

                    List<Document> documentList = customPool.submit(() -> coreDocument.sentences()
                                    .parallelStream()
                                    .map(sentence -> new Document(sentence.text()))
                                    .toList())
                            .get();

                    log.info("{} {} sentences found", pdfResource.getFilename(), documentList.size());
                    AtomicInteger counter = new AtomicInteger(0);
                    customPool.submit(() -> chunkList(documentList, 100).parallelStream()
                                    .forEach(docs -> {
                                        vectorStore.accept(docs);
                                        log.info("{} Added {} sentences", pdfResource.getFilename(), counter.addAndGet(docs.size()));
                                    }))
                            .get();
                } catch (IOException | InterruptedException | ExecutionException e) {
                    log.error("Unable to process", e);
                }
            });
        } catch (Exception e) {
            log.error("Huh?", e);
        } finally {
            customPool.close();
        }
    }

    <T> List<List<T>> chunkList(List<T> inputList, int chunkSize) {
        List<List<T>> chunks = new ArrayList<>();
        for (int i = 0; i < inputList.size(); i += chunkSize) {
            chunks.add(new ArrayList<>(inputList.subList(i, Math.min(inputList.size(), i + chunkSize))));
        }
        return chunks;
    }
}
