package com.tjjhtjh.memorise.domain.memo.controller;

import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.memo.service.MemoService;
import com.tjjhtjh.memorise.domain.memo.service.dto.request.MemoRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RequiredArgsConstructor
@RequestMapping("/memos")
@RestController
public class MemoController {

    private final MemoService memoService;

    @PostMapping
    public ResponseEntity<Object> registMemo(@RequestBody MemoRequest memoRequest){
        memoService.createMemo(memoRequest);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @PutMapping("/{memoId}")
    public ResponseEntity<Object> updateMemo(@PathVariable Long memoId, @RequestBody MemoRequest memoRequest) throws MemoException {
        memoService.updateMemo(memoRequest, memoId);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @DeleteMapping("/{memoId}")
    public ResponseEntity<Object> deleteMemo(@PathVariable Long memoId,@RequestBody MemoRequest memoRequest) throws MemoException {
        memoService.fakeDeleteMemo(memoId,memoRequest);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @PostMapping("/{memoId}/bookmarks/{email}")
    public ResponseEntity<?> addBookmark(@PathVariable Long memoId,@PathVariable String email) throws MemoException {
        memoService.addBookmark(memoId,email);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @DeleteMapping("/{memoId}/bookmarks/{email}")
    public ResponseEntity<?> deleteBookmark(@PathVariable Long memoId , @PathVariable String email) throws MemoException {
        memoService.deleteBookmark(memoId,email);
        return new ResponseEntity<>(HttpStatus.OK);
    }
}
