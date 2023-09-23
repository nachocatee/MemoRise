package com.tjjhtjh.memorise.domain.memo.controller;

import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.memo.service.MemoService;
import com.tjjhtjh.memorise.domain.memo.service.dto.request.MemoRequest;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import com.tjjhtjh.memorise.global.file.service.AwsS3Service;
import com.tjjhtjh.memorise.global.file.service.dto.CreateFileRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Slf4j
@RequiredArgsConstructor
@RequestMapping("/memos")
@RestController
public class MemoController {

    private final MemoService memoService;
    private final AwsS3Service awsS3Service;
    private static String dirName = "memo-image";

    @PostMapping("/upload")
    public ResponseEntity<CreateFileRequest> uploadMultipleFile(@RequestPart(required = false) MultipartFile file) {
        return ResponseEntity.ok(awsS3Service.uploadMultiFile(file, dirName));
    }

    @PostMapping("/{itemSeq}")
    public ResponseEntity<Object> registMemo(@PathVariable Long itemSeq, @RequestBody MemoRequest memoRequest){
        memoService.createMemo(memoRequest, itemSeq);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @PutMapping("/{memoId}/{itemSeq}")
    public ResponseEntity<Object> updateMemo(
            @PathVariable Long memoId, @RequestBody MemoRequest memoRequest, @PathVariable Long itemSeq
    ) throws MemoException {
        memoService.updateMemo(memoRequest, memoId,itemSeq);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @DeleteMapping("/{memoId}/{itemSeq}")
    public ResponseEntity<Object> deleteMemo(
            @PathVariable Long memoId,@RequestBody MemoRequest memoRequest , @PathVariable Long itemSeq
    ) throws MemoException {
        memoService.fakeDeleteMemo(memoId,memoRequest,itemSeq);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @PostMapping("/{memoId}/bookmarks/{userSeq}")
    public ResponseEntity<?> addBookmark(@PathVariable Long memoId,@PathVariable Long userSeq) throws MemoException {
        memoService.addBookmark(memoId,userSeq);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @DeleteMapping("/{memoId}/bookmarks/{userSeq}")
    public ResponseEntity<?> deleteBookmark(@PathVariable Long memoId , @PathVariable Long userSeq) {
        memoService.deleteBookmark(memoId,userSeq);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @GetMapping("/{itemSeq}/list/{userSeq}")
    public ResponseEntity<?> myMemoListOfItem(@PathVariable Long itemSeq ,@PathVariable Long userSeq){
        return new ResponseEntity<>(memoService.itemMemoView(itemSeq,userSeq),HttpStatus.OK);
    }

    @GetMapping("/{memoId}/{userSeq}")
    public ResponseEntity<?> detailMemo(@PathVariable Long memoId, @PathVariable Long userSeq) throws MemoException {
        return new ResponseEntity<>(memoService.detailMemo(memoId), HttpStatus.OK);
    }
}
