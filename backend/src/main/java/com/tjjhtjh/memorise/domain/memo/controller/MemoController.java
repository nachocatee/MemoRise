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
}
